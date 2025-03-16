from typing import List
from typing import TYPE_CHECKING
import torch
import asyncio
from sentence_transformers import SentenceTransformer, util

if TYPE_CHECKING:
    from main import PostData, QuestionApiDTO
    
class SpamDetector:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS", threshold=0.8):
        print("모델 로딩 중...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.threshold = threshold
        self.db_posts = {}
        self.model_name = model_name
        print("모델 로딩 완료!")
        
    async def init(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
    async def async_encode(self, text: str):
        # CPU 집약적 인코딩 작업을 별도의 쓰레드에서 실행
        return await asyncio.to_thread(self.model.encode, text, convert_to_tensor=True, device=self.device)

    async def async_find_most_relevant_base_id(self, new_emb, questions: List["QuestionApiDTO"]):
        # CPU 집약적 작업을 별도 쓰레드에서 실행
        return await asyncio.to_thread(self.find_most_relevant_base_id, new_emb, questions)

    def find_most_relevant_base_id(self, new_emb, questions: List["QuestionApiDTO"]):
        valid_candidates = []
        for question in questions:
            # 동기적으로 인코딩하는 대신, 이미 동기 함수인 encode를 사용
            existing_emb = self.model.encode(question.title + " " + question.content, 
                                               convert_to_tensor=True, device=self.device)
            #코사인 유사도 계산 후 임계값보다 높으면 저장
            sim = util.cos_sim(new_emb, existing_emb).item()
            if sim >= self.threshold:
                valid_candidates.append(question.id)
        if not valid_candidates:
            return None  # 유사한 기준 글이 없으면 새로운 기준으로 등록
        valid_candidates.sort(key=lambda x: x)
        return valid_candidates

    async def async_check_spam_and_store(self, post_data: "PostData", questions: List["QuestionApiDTO"]) -> int:
        # 게시글 데이터에서 제목, 내용 추출
        new_text = post_data.title.strip() + " " + post_data.content.strip()
        #텍스트 임베딩
        new_emb = await self.async_encode(new_text)
        most_similar_id = await self.async_find_most_relevant_base_id(new_emb, questions)
        if most_similar_id is None:
            return "도배아님"
        else:
            print(f"게시글이 기존 질문 {most_similar_id}과 유사")
            print(most_similar_id)
            if len(most_similar_id) >= 2:
                return "도배"
            else : return "도배아님"