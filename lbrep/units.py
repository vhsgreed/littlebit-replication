"""Registered models, matrix enumeration, exclusions, and the seed-variance subset."""
import hashlib, re

# Run order is registered: if the hard deadline hits, models are scored in this order only.
MODELS = [
    ('llama2-7b', 'NousResearch/Llama-2-7b-hf', '8efe6c9b93655b934e27bd9981e3ec13e55aee9d'),
    ('llama3-8b', 'NousResearch/Meta-Llama-3-8B', '315b20096dc791d381d514deb5f8bd9c8d6d3061'),
    ('qwen3-8b', 'Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
    ('mistral-7b-v0.3', 'mistralai/Mistral-7B-v0.3', 'caa1feb0e54d415e2df31207e5f4e273e33509b1'),
    ('olmo2-7b', 'allenai/OLMo-2-1124-7B', '7df9a82518afdecae4e8c026b27adccc8c1f0032'),
    ('gemma3-4b', 'unsloth/gemma-3-4b-pt', '52aba93981c6ad7712b030eb6dd496ece1d279d6'),
]
BY_KEY = {k: (r, s) for k, r, s in MODELS}

BPPS = [1.0, 0.55, 0.3, 0.1]
INT_BITS = [8, 4, 3, 2]
VAR_SEEDS = [1, 2, 3, 4]

PROJ = re.compile(r'^(?:model\.|language_model\.model\.|model\.language_model\.)?layers\.(\d+)\.'
                  r'(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$')

# Pilot matrices: outcome values for these were seen before the freeze (PLAN.md change log).
# Whole layers are excluded so no seen matrix leaks into a confirmatory count.
PILOT_LAYERS = {('llama2-7b', 2), ('llama2-7b', 15)}


def matrices(names):
    out = []
    for n in names:
        m = PROJ.match(n)
        if m:
            out.append((int(m.group(1)), m.group(2).split('.')[1], n))
    return sorted(out)


def is_pilot(model, layer):
    return (model, layer) in PILOT_LAYERS


def in_var_subset(model, name):
    """Registered 10% subset for extra seeds: sha256 of 'model|tensor name' mod 10 == 0."""
    return int(hashlib.sha256(f'{model}|{name}'.encode()).hexdigest(), 16) % 10 == 0
