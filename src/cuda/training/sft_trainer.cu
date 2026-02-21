#include "sft_trainer.cuh"

namespace quila {

__host__ void update_sft_config(SFTConfig* config, int step) {
    config->current_step = step;

    // Disable KFE and topology evolution during first 10% (Req 16.1.1)
    if (is_warmup_phase(config)) {
        config->kfe_enabled = false;
        config->topology_evolution_enabled = false;
    } else {
        config->kfe_enabled = true;
        config->topology_evolution_enabled = true;
    }
}

} // namespace quila
