# Open-source asset audit (all-tier transfer phase)

Systematic search of GitHub/Hugging Face for pretrained ViZDoom agents, demonstration data,
scenario generators and imitation/recurrent training implementations. Full prior audit lives in
`docs/reuse-and-transfer-audit.md`; this file records the new phase's decisions.

| Asset | URL | Licence | Artifact | Class | Expected tier benefit | Decision |
|---|---|---|---|---|---|---|
| 3D Control Deep RL | github.com/edbeeching/3d_control_deep_rl | MIT (verified) | 16 `.pth.tar`, ~1.59 MB each | 1 Direct (encoder) | All tiers via visual features | **Already integrated** into champion |
| doomfly-rl | huggingface.co/fabiononato/doomfly-rl | MIT | safetensors, 27–48 M params, 22 actions | 5 Reject | Combat teacher only | Too large / incompatible action head |
| ViZDoomBot stable-baselines-agent | github.com/ViZDoomBot/stable-baselines-agent | **none found** | LFS TF1 zips | 5 Reject | — | Legal risk + obsolete stack |
| callumhay/vizdoom_ppo_rnd | github.com/callumhay/vizdoom_ppo_rnd | MIT | **no weights found** | 4 Inspiration | Long-horizon failure note | No code reused |
| hyyh619/ViZDoom-Imitation-Defend-Center | github.com/hyyh619/ViZDoom-Imitation-Defend-Center | MIT | code only (Keras 1.x/TF 0.12), no weights | 4 Inspiration | Recurrent imitation idea | Reuse idea, not code |
| NavDoom | github.com/agiantwhale/NavDoom | MIT | maze generator, no weights | 3 Scenario generation | Tier 2 diversity | Idea only; our own generator already verified |
| gameNgen ViZDoom datasets | huggingface.co/arnaudstiegler/... | none stated | 187 MB / 48 GB episodes | 5 Reject | Combat/door data | Licence unclear, huge, format mismatch |
| p-doom/doom-dataset | CC0 | 10M frames, 72 GB | 5 Reject | Coverage | Too large; no compatible action labels |
| Sample Factory HF models | huggingface.co/edbeeching/doom_* | none stated | .pth checkpoints | 5 Reject | — | Legal cloud, scenario-locked |

**Search result for door/temporal interaction:** no compatible external dataset or pretrained
policy specifically for ViZDoom door interaction with our observation/action contract was
found. The verified Tier-2 bottleneck therefore requires our own targeted trajectory
collection and sequence-aware training.

**Chosen highest-value integration:** sequence-level behavior cloning with a recurrent
KEYSTONE student (`transfer/seq_bc.py`) trained on our own oracle demonstrations with episode
IDs, migrating all champion weights except the new GRU. This directly targets "press USE once,
then wait for door to open", which a reactive student cannot represent.

Licences preserved: `transfer/licenses/EDBEECHING_LICENSE.txt` (MIT, Copyright 2019 Ed Beeching).
