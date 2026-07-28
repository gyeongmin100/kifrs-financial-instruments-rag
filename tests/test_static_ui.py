import re
import unittest
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "src" / "accounting_rag" / "api" / "static"


class StaticUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC / "index.html").read_text(encoding="utf-8")
        cls.css = (STATIC / "styles.css").read_text(encoding="utf-8")
        cls.javascript = (STATIC / "app.js").read_text(encoding="utf-8")

    def test_assets_and_api_contract_are_wired(self):
        self.assertIn('href="/static/styles.css"', self.html)
        self.assertIn('src="/static/app.js"', self.html)
        self.assertIn('apiUrl("/v1/jobs")', self.javascript)
        self.assertIn("sessionStorage", self.javascript)
        for field in ("question", "images", "top_k", "debug", "request_id"):
            self.assertIn(field, self.javascript)

    def test_question_input_is_limited_to_2000_characters(self):
        self.assertIn('maxlength="2000"', self.html)

    def test_removed_navigation_and_advanced_controls_stay_removed(self):
        for value in ("app-header", "privacy-note", "검색 설정", "question-count"):
            self.assertNotIn(value, self.html)

    def test_local_chat_history_sidebar_exists(self):
        self.assertIn('id="sidebar"', self.html)
        self.assertIn('id="new-chat-button"', self.html)
        self.assertIn('id="history-list"', self.html)
        self.assertIn("localStorage", self.javascript)
        self.assertIn("HISTORY_KEY", self.javascript)
        self.assertIn("deleteChat", self.javascript)

    def test_accessibility_and_safe_text_rendering(self):
        self.assertIn('lang="ko"', self.html)
        self.assertIn('aria-live="polite"', self.html)
        self.assertIn('role="alert"', self.html)
        self.assertNotIn("innerHTML", self.javascript)
        self.assertGreaterEqual(len(re.findall(r"<label\b", self.html)), 1)

    def test_internal_evidence_metadata_is_hidden(self):
        for value in (
            "candidate_source",
            "pdf_page_start",
            "pdf_page_end",
            "graph_hop",
            "graph_path",
            "GraphRAG",
            "Hybrid RAG",
        ):
            self.assertNotIn(value, self.javascript)
        self.assertIn("evidenceStatement", self.javascript)

    def test_responsive_and_reduced_motion_styles_exist(self):
        # 확대 배율을 적용해도 뷰포트 분기점은 그대로여야 반응형이 유지된다.
        self.assertIn("@media(max-width:800px)", self.css)
        self.assertIn("@media(max-height:680px) and (min-width:801px)", self.css)
        self.assertIn("prefers-reduced-motion", self.css)

    def test_scroll_position_survives_reload(self):
        self.assertIn("SCROLL_KEY", self.javascript)
        self.assertIn("restoreScroll", self.javascript)
        self.assertIn('window.history.scrollRestoration="manual"', self.javascript)
        # 복원은 즉시 적용해야 맨 아래로 끌려갔다 돌아오는 움직임이 안 보인다.
        self.assertIn('behavior:"instant"', self.javascript)
        self.assertIn("if(!restoreScroll(id))scrollToLatest()", self.javascript)

    def test_chat_landing_and_image_attachment_controls_exist(self):
        self.assertIn("금융상품 AI Accountant에 오신 걸 환영합니다", self.html)
        self.assertIn('id="image-input"', self.html)
        self.assertIn('accept="image/png,image/jpeg,image/webp,image/gif"', self.html)
        self.assertIn('addEventListener("paste"', self.javascript)
        self.assertIn('addEventListener("drop"', self.javascript)
        self.assertIn("indexedDB", self.javascript)
        self.assertIn("saveRetryPayload", self.javascript)
        self.assertIn("loadRetryPayload", self.javascript)

    def test_actionable_http_errors_skip_the_retry_button(self):
        self.assertIn("renderHttpError", self.javascript)
        self.assertIn("요청이 많습니다. 잠시 후 다시 시도해 주세요.", self.javascript)
        self.assertIn("질문이나 첨부 이미지를 다시 확인해 주세요.", self.javascript)
        # 429·422 안내는 재시도 버튼 없이 문구만 그린다.
        guidance = re.search(r"const renderHttpError=.*?\};", self.javascript, re.S)
        self.assertIsNotNone(guidance)
        self.assertNotIn("retry-button", guidance.group(0))

    def test_pending_copy_and_avatar_presentation(self):
        self.assertIn("기준서에서 근거를 찾고 있습니다.", self.javascript)
        self.assertNotIn("기준서의 연결 관계와 근거를 찾고 있습니다", self.javascript)
        self.assertIn('AVATAR_SRC="/static/favicon.png"', self.javascript)
        self.assertIn("copyButton", self.javascript)
        self.assertIn("answerToText", self.javascript)
        self.assertIn(".copy-button", self.css)

    def test_sidebar_and_active_chat_survive_reload(self):
        self.assertIn("SIDEBAR_KEY", self.javascript)
        self.assertIn("ACTIVE_KEY", self.javascript)
        self.assertIn("applySidebarState", self.javascript)
        # 저장된 값이 없으면 닫힘이 기본이라 첫 랜딩에서 사이드바가 닫혀 있다.
        self.assertIn('readStore(localStorage,SIDEBAR_KEY)==="1")openSidebar(false);else closeSidebar(false)', self.javascript)
        # 첫 페인트부터 닫혀 있어야 사이드바가 열렸다 닫히는 모션이 보이지 않는다.
        self.assertIn('<body class="sidebar-collapsed sidebar-boot">', self.html)
        self.assertIn(".sidebar-boot .sidebar{transition:none}", self.css)
        self.assertIn('classList.remove("sidebar-boot")', self.javascript)

    def test_navigation_stays_available_while_a_job_runs(self):
        # 진행 중이어도 대화 이동과 새 대화가 막히지 않아야 한다.
        self.assertNotIn("답변이 완료된 후 다른 대화를 열 수 있습니다", self.javascript)
        self.assertNotIn("답변이 완료된 후 새 대화를 시작할 수 있습니다", self.javascript)
        # 결과는 보고 있는 화면이 아니라 요청을 시작한 대화에 기록한다.
        self.assertIn("storeMessageTo(chatId,\"assistant\"", self.javascript)
        self.assertIn("job.chatId", self.javascript)

    def test_reload_preserves_the_view_while_a_job_runs(self):
        self.assertIn("restoreView(viewedChatId)", self.javascript)
        self.assertNotIn("setActiveChat(chatId);const token", self.javascript)

    def test_image_only_message_has_no_generated_text(self):
        self.assertNotIn("첨부 이미지의 내용을 분석해 주세요.", self.javascript)
        self.assertIn('if(text)content.append(make("div","",text))', self.javascript)

    def test_favicon_points_at_a_real_static_file(self):
        self.assertIn('href="/static/favicon.png"', self.html)
        self.assertNotIn('href="/favicon.ico"', self.html)
        self.assertLess((STATIC / "favicon.png").stat().st_size, 100_000)


if __name__ == "__main__":
    unittest.main()
