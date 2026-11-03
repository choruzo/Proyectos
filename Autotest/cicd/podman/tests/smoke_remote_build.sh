#!/bin/bash
# =============================================================================
# smoke_remote_build.sh - Smoke test de la compilacion delegada al host
# =============================================================================
# Se ejecuta DENTRO de la imagen del orquestador y simula el host con un
# 'ssh' y un 'build-wrapper' falsos. Valida:
#   config -> compile.sh prepare -> compile.sh build (CICD_BUILD_MODE=ssh)
#          -> build_runner.sh -> wrapper -> compile.sh validate
#
# Uso:
#   docker run --rm --entrypoint /bin/bash localhost/cicd-orchestrator:local \
#     -lc 'bash /home/agent/cicd/podman/tests/smoke_remote_build.sh'
# =============================================================================

set -euo pipefail

SIM=/tmp/cicd-sim
rm -rf "$SIM"
mkdir -p "$SIM/bin" "$SIM/config" "$SIM/db" "$SIM/logs" \
         "$SIM/repo/Development_TTCF/ttcf/utils/dvds"

cp /home/agent/cicd/config/ci_cd_config.yaml "$SIM/config/ci_cd_config.yaml"
yq -i '.general.working_directory="/tmp/cicd-sim"
       | .general.log_directory="/tmp/cicd-sim/logs"
       | .general.db_path="/tmp/cicd-sim/db/pipeline.db"
       | .git.repo_local_path="/tmp/cicd-sim/repo"
       | .git.compile_path="/tmp/cicd-sim/compile"
       | .compilation.build_script="Development_TTCF/ttcf/utils/dvds/build_DVDs.sh"
       | .compilation.output_iso="fake.iso"
       | .compilation.timeout_seconds=30' \
   "$SIM/config/ci_cd_config.yaml"

sqlite3 "$SIM/db/pipeline.db" < /home/agent/cicd/db/init_db.sql

cat > "$SIM/repo/Development_TTCF/ttcf/utils/dvds/build_DVDs.sh" <<'BUILD'
#!/bin/bash
echo "[fake build] compiling in $(pwd)"
sleep 1
echo fake-iso > fake.iso
BUILD

cat > "$SIM/wrapper" <<'WRAP'
#!/bin/bash
while [ "${1:-}" != "--out-dir" ]; do shift; done
shift; out=$1; shift
mkdir -p "$out"
echo "[fake wrapper] out-dir=$out"
exec "$@"
WRAP

cat > "$SIM/bin/ssh" <<'SSH'
#!/bin/bash
# Simula ssh: ejecuta el ultimo argumento (el comando remoto) en local.
for last; do :; done
exec bash -c "$last"
SSH

chmod +x "$SIM/repo/Development_TTCF/ttcf/utils/dvds/build_DVDs.sh" \
         "$SIM/wrapper" "$SIM/bin/ssh"

export PATH="$SIM/bin:$PATH"
export CICD_HOME="$SIM"
export CICD_BUILD_MODE=ssh
export CICD_BUILD_SSH=fake@127.0.0.1
export CICD_BUILD_RUNNER=/home/agent/cicd/podman/host/build_runner.sh
export CICD_BUILD_WRAPPER="$SIM/wrapper"
export CICD_BUILD_SSH_KEY=/nonexistent

cd /home/agent/cicd

echo "=== prepare ==="
./scripts/compile.sh prepare

echo "=== build (delegado) ==="
./scripts/compile.sh build

echo "=== validate ==="
./scripts/compile.sh validate

echo ""
echo "SMOKE TEST OK: compilacion delegada y validada"
