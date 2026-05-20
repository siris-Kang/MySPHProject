# 커밋 메시지 자동화
- `commit-msg` 훅이 커밋 메시지의 prefix 포맷을 자동으로 정리하고, Jira 키가 있으면 본문에 추가한다.
- `docs: 제목`, `TEST: 제목`, `[docs] 제목`, `[Test] 제목`, `✅ [TEST] 제목` 모두 같은 형식으로 변환된다.

<br>

# 설정 방법
git pull 후 **레포 루트**에서 딱 한 번만 실행한다.
```
git config core.hooksPath .githooks
git config --get core.hooksPath
```
두번째 줄 실행 시 .githooks가 나오면 설정 완료  