#ifndef BACKENDS_WAVE_H
#define BACKENDS_WAVE_H

#include "backends/base.h"

struct WaveBackendFactory final : public BackendFactory {
public:
    bool init() override;

    bool querySupport(BackendType type) override;

    std::string probe(BackendType type) override;

    BackendPtr createBackend(ALCdevice *device, BackendType type) override;

    static BackendFactory &getFactory();
};

#endif /* BACKENDS_WAVE_H */
