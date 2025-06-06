# Project Cleanup Plan

## Files to Move/Organize

### Docker Files (Move to `docker/`)
- [x] Dockerfile.nvidia → Already consolidated in docker/
- [ ] docker-compose.nvidia-launchable.yml → Move to docker/docker-compose.launchable.yml

### Deployment Scripts (Move to `scripts/deployment/`)
- [ ] deploy.sh
- [ ] deploy_all.sh
- [ ] deploy_t4_app.sh
- [ ] deploy_t4_backend.sh
- [ ] deploy_to_jupyter_vm.sh
- [ ] deploy_to_vm.sh
- [ ] deploy_vercel_frontend.sh
- [ ] setup_nvidia_backend.sh
- [ ] setup_vercel_frontend.sh

### Environment Files (Move to `config/env/`)
- [ ] .env.example
- [ ] .env.frontend.sap.prod
- [ ] .env.frontend.vercel.prod
- [ ] .env.nvidia.prod
- [ ] .env.nvidia.t4.prod
- [ ] .env.test
- [ ] .env.together.prod
- [ ] .env.vercel

### Documentation (Consolidate in `docs/`)
- [ ] BACKEND_IMPROVEMENTS.md
- [ ] DEPLOYMENT.md
- [ ] EXTENSIBILITY.md
- [ ] IMPROVEMENTS.md
- [ ] README_DEPLOYMENT.md
- [ ] TESTING.md
- [ ] VERCEL_500_QUICKSTART.md

### Testing Files (Move to `tests/`)
- [ ] test_app.py
- [ ] test_app_enhanced.py
- [ ] test_config.json
- [ ] test_results.json
- [ ] test_run.log
- [ ] run_automated_tests.py
- [ ] run_tests.sh
- [ ] run_cpu_local.sh
- [ ] load_test.py
- [ ] create_test_data.py

### Utility Scripts (Move to `scripts/utils/`)
- [ ] archive_deprecated_files.sh
- [ ] cleanup-command.sh
- [ ] move_to_new_structure.sh

### Vercel Configuration (Move to `config/vercel/`)
- [ ] vercel.frontend.json
- [ ] vercel.json
- [ ] vercel.json.bak
- [ ] .vercelignore

### NVIDIA/Launchable Files (Move to `config/nvidia/`)
- [ ] nvidia-launchable.yaml
- [ ] nvidia_launchable.ipynb
- [ ] build_launchable.sh

## Retain in Root Directory

### Essential Files
- [ ] README.md (update with new structure)
- [ ] LICENSE
- [ ] CONTRIBUTING.md
- [ ] REUSE.toml
- [ ] VERSION
- [ ] Makefile
- [ ] pyproject.toml
- [ ] poetry.lock
- [ ] requirements.txt
- [ ] package.json
- [ ] .gitignore
- [ ] .pre-commit-config.yaml
- [ ] app.py (main application entry point)

## Cleanup Tasks

1. Create necessary directories
2. Move files to appropriate locations
3. Update import paths and references if needed
4. Update documentation to reflect new structure
5. Create a comprehensive README with the new project structure