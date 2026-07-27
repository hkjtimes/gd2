"""PPO (Proximal Policy Optimization), написанный с нуля на numpy.

Градиенты по клипованной surrogate-функции, value-loss и энтропии
выведены аналитически и прогоняются через ручной backprop модели.
"""

import numpy as np

from ..nn.layers import softmax


class PPOConfig:
    def __init__(self, n_steps=512, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=4, n_minibatches=8, vf_coef=0.5, ent_coef=0.01,
                 max_grad_norm=0.5):
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.n_minibatches = n_minibatches
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm


class PPO:
    def __init__(self, model, vec_env, config: PPOConfig | None = None, seed=0):
        self.model = model
        self.venv = vec_env
        self.cfg = config or PPOConfig()
        self.rng = np.random.default_rng(seed)
        self.obs = vec_env.reset()
        # статистика для мониторинга
        self.ep_returns = []
        self.ep_progress = []
        self.ep_wins = []
        self._running_ret = np.zeros(vec_env.n)

    # ---------------------------------------------------------------- rollout

    def collect_rollout(self):
        cfg, venv = self.cfg, self.venv
        T, N = cfg.n_steps, venv.n
        obs_buf = np.zeros((T, N, self.obs.shape[1]), dtype=np.float32)
        act_buf = np.zeros((T, N), dtype=np.int64)
        logp_buf = np.zeros((T, N))
        rew_buf = np.zeros((T, N))
        done_buf = np.zeros((T, N), dtype=bool)
        val_buf = np.zeros((T, N))

        for t in range(T):
            actions, logp, values = self.model.act(self.obs, self.rng)
            obs_buf[t] = self.obs
            act_buf[t] = actions
            logp_buf[t] = logp
            val_buf[t] = values
            self.obs, rewards, dones, infos = venv.step(actions)
            rew_buf[t] = rewards
            done_buf[t] = dones
            self._running_ret += rewards
            for i, d in enumerate(dones):
                if d:
                    self.ep_returns.append(self._running_ret[i])
                    self.ep_progress.append(infos[i]["progress"])
                    self.ep_wins.append(float(infos[i]["won"]))
                    self._running_ret[i] = 0.0

        _, last_values = self.model.forward(self.obs)

        # GAE-Lambda
        adv = np.zeros((T, N))
        gae = np.zeros(N)
        for t in range(T - 1, -1, -1):
            next_val = last_values if t == T - 1 else val_buf[t + 1]
            mask = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_val * mask - val_buf[t]
            gae = delta + cfg.gamma * cfg.lam * mask * gae
            adv[t] = gae
        returns = adv + val_buf

        flat = lambda a: a.reshape(T * N, *a.shape[2:])
        return (flat(obs_buf), flat(act_buf), flat(logp_buf),
                flat(adv), flat(returns))

    # ----------------------------------------------------------------- update

    def update(self, batch):
        cfg = self.cfg
        obs, actions, old_logp, adv, returns = batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        idx = np.arange(n)
        mb_size = n // cfg.n_minibatches
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for _ in range(cfg.epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                if len(mb) < 2:
                    continue
                self._update_minibatch(obs[mb], actions[mb], old_logp[mb],
                                       adv[mb], returns[mb], stats)
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)
        return stats

    def _update_minibatch(self, obs, actions, old_logp, adv, returns, stats):
        cfg, model = self.cfg, self.model
        B = len(obs)
        logits, values = model.forward(obs)
        probs = softmax(logits)
        p_act = probs[np.arange(B), actions]
        logp = np.log(p_act + 1e-10)
        ratio = np.exp(logp - old_logp)

        # --- policy loss (клипованный surrogate) ---
        clipped = (ratio > 1 + cfg.clip) | (ratio < 1 - cfg.clip)
        surr1 = ratio * adv
        surr2 = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
        active = surr1 <= surr2 + 1e-12       # градиент течёт там, где min = surr1
        dlogp = np.where(active, -adv * ratio, 0.0) / B

        # --- entropy bonus: L -= ent_coef * H;  dH/dlogits через softmax ---
        logp_all = np.log(probs + 1e-10)
        entropy = -(probs * logp_all).sum(axis=1)
        # d(-H)/dlogits = probs * (logp_all + H[:, None]) -- по формуле softmax
        d_ent = probs * (logp_all + entropy[:, None]) * (cfg.ent_coef / B)

        # градиент по логитам от policy-части: dlogp/dlogits = onehot - probs
        grad_logits = probs * dlogp[:, None]
        grad_logits[np.arange(B), actions] -= dlogp
        grad_logits = -grad_logits            # знак: dlogp_dlogits = onehot - probs
        grad_logits += d_ent

        # --- value loss 0.5 * (v - R)^2 ---
        v_err = values - returns
        grad_values = cfg.vf_coef * v_err / B

        model.opt.zero_grad()
        model.backward_heads(grad_logits, grad_values)
        model.opt.step(cfg.max_grad_norm)

        stats["pi_loss"] += float(-np.minimum(surr1, surr2).mean())
        stats["v_loss"] += float(0.5 * (v_err ** 2).mean())
        stats["entropy"] += float(entropy.mean())
        stats["clip_frac"] += float(clipped.mean())

    # ------------------------------------------------------------------- train

    def train_iteration(self):
        batch = self.collect_rollout()
        return self.update(batch)

    def recent_stats(self, window=50):
        if not self.ep_returns:
            return {"ret": 0.0, "progress": 0.0, "win_rate": 0.0, "episodes": 0}
        r = self.ep_returns[-window:]
        p = self.ep_progress[-window:]
        w = self.ep_wins[-window:]
        return {"ret": float(np.mean(r)), "progress": float(np.mean(p)),
                "win_rate": float(np.mean(w)), "episodes": len(self.ep_returns)}
