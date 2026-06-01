#!/usr/bin/env bash
# Проверка рабочей копии и истории Git на типичные утечки секретов.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FAIL=0
GREP_EXCLUDES=(
  --exclude-dir=.git
  --exclude-dir=__pycache__
  --exclude=check-secrets.sh
  --exclude=SECURITY.md
  --exclude=.env_example
  --exclude=.env
)

echo "==> Проверка рабочей копии..."

# API-ключи в коде
if grep -rE 'sk-[A-Za-z0-9]{20,}' "${GREP_EXCLUDES[@]}" . 2>/dev/null \
  | grep -v 'sk-your-api-key-here'; then
  echo "ОШИБКА: найден похожий на API-ключ (sk-...)"
  FAIL=1
fi

# Fallback с ключом в os.getenv
if grep -rE 'getenv\([^)]*,\s*["\x27]sk-' "${GREP_EXCLUDES[@]}" . 2>/dev/null; then
  echo "ОШИБКА: захардкоженный sk- в os.getenv(..., default)"
  FAIL=1
fi

# Длинный hex secret_key (кроме placeholder в searxng)
if grep -rE 'secret_key:\s*["\x27][a-f0-9]{32,}["\x27]' "${GREP_EXCLUDES[@]}" . 2>/dev/null \
  | grep -v 'change-me-in-env'; then
  echo "ОШИБКА: похоже на реальный secret_key в конфиге"
  FAIL=1
fi

# Захардкоженные ключи в docker-compose (должно быть ${VAR})
if grep -E 'API_KEY=|OPENAI_API_KEYS=' docker-compose.yml 2>/dev/null \
  | grep -vE '\$\{[A-Z_]+\}'; then
  echo "ОШИБКА: в docker-compose.yml ключ не через переменную окружения"
  FAIL=1
fi

if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "ОШИБКА: файл .env отслеживается Git — удалите из индекса: git rm --cached .env"
  FAIL=1
fi

echo "==> Проверка истории Git (все коммиты)..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git grep -E 'sk-[A-Za-z0-9]{20,}' "$(git rev-list --all)" -- . \
      ':!scripts/check-secrets.sh' ':!.env_example' 2>/dev/null \
    | grep -v 'sk-your-api-key-here'; then
    echo "ОШИБКА: sk-... в истории коммитов"
    FAIL=1
  fi

  if git grep -E 'getenv\([^)]*,\s*["\x27]sk-' "$(git rev-list --all)" -- . \
      ':!scripts/check-secrets.sh' ':!.env_example' 2>/dev/null; then
    echo "ОШИБКА: захардкоженный sk- в истории"
    FAIL=1
  fi

  if git grep -E 'secret_key:\s*["\x27][a-f0-9]{32,}["\x27]' "$(git rev-list --all)" -- . \
      ':!scripts/check-secrets.sh' ':!.env_example' 2>/dev/null \
    | grep -v 'change-me-in-env'; then
    echo "ОШИБКА: secret_key в истории"
    FAIL=1
  fi
fi

if [[ "$FAIL" -eq 0 ]]; then
  echo "OK: подозрительных секретов не найдено."
  exit 0
fi

exit 1
