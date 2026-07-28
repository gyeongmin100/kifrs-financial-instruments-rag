// 배포 환경별 설정. apiBase가 빈 문자열이면 같은 서버의 API를 사용한다.
// Cloudflare Pages 등 프론트엔드 분리 배포 시 이 값만 백엔드 주소로 바꾼다.
window.APP_CONFIG = { apiBase: "" };
