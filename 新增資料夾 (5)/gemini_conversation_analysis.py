# -*- coding: utf-8 -*-
"""
Gemini 自殺危機對話分析程式
混合方法分析: 質性 + 量化
"""

import os
import re
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 可選: Gemini API
try:
    import google.generativeai as genai
except ImportError:  # 未安裝時不中斷
    genai = None

# 可選: 詞雲
try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class GeminiConversationAnalyzer:
    """Gemini 對話分析器"""
    
    def __init__(self, file_path, use_gemini_api=False):
        self.file_path = file_path
        self.use_gemini_api = use_gemini_api
        self.content = self._load_file()
        
        # 若啟用 Gemini API,初始化模型(沒有金鑰則回退)
        self.gemini_model = self._init_gemini_model() if use_gemini_api else None
        
        # 解析對話 (需要在 gemini_model 初始化後)
        self.turns = self._parse_turns()
        
        # 情緒詞典 (可擴充)
        self.emotion_dict = {
            '負面高強度': ['想死', '自殺', '痛苦', '絕望', '崩潰', '無助', '悲傷', '難過', '累', '恐懼', '害怕', '焦慮', '緊張'],
            '負面中強度': ['不舒服', '難受', '不安', '擔心', '煩惱', '困擾', '疲憊'],
            '中性/過渡': ['空空的', '發呆', '呼吸', '存在', '看著', '等待'],
            '正面低強度': ['平靜', '放鬆', '溫暖', '舒服', '安全'],
            '正面高強度': ['愛', '快樂', '感動', '力量', '勇氣', '希望']
        }
        
        # 療癒策略詞典
        self.healing_strategies = {
            '正念覺察': ['覺察', '覺知', '同在', '當下', '呼吸', '感受', '觀察', '看著'],
            '自我慈悲': ['允許', '接納', '陪伴', '溫柔', '照顧自己', '疼自己'],
            '身體接地': ['吃麵', '煮麵', '海邊', '外木山', '靠著牆', '坐著', '石頭'],
            '認知重構': ['可能錯了', '換個角度', '不是真的', '理解'],
            '情緒釋放': ['哭', '眼淚', '流出來', '釋放'],
            '轉移注意': ['看新聞', '聽音樂', '看海', '散步'],
            '社會支持': ['抱抱', '陪伴', '一起', '支持']
        }
        
        # 風險與保護因子
        self.risk_factors = ['想死', '自殺', '結束', '沒人愛', '一個人', '孤獨', '無助', '絕望']
        self.protective_factors = ['二姊', '三姊', '姪子', '姪女', '責任', '照顧自己', '呼吸', '存在']
    
    def _load_file(self):
        """載入文本檔案"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _init_gemini_model(self):
        """初始化 Gemini 模型 (若環境變數缺失則回傳 None)"""
        if genai is None:
            return None
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return None
        try:
            genai.configure(api_key=api_key)
            # 使用輕量模型降低成本
            return genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            return None
    
    def _classify_speaker_via_gemini(self, text):
        """呼叫 Gemini 判斷發話者 (User/Gemini)。失敗時回傳 None。"""
        if not self.gemini_model:
            return None
        prompt = (
            "你是對話標註器，判斷這段文字是誰說的。"\
            "只回答 User 或 Gemini 兩個字。"\
            "User 通常描述自己的感受、行動或問題；Gemini 會給建議、安撫、分析、列點。"\
            "輸入：" + text[:4000]
        )
        try:
            resp = self.gemini_model.generate_content(prompt)
            label = resp.text.strip()
            if 'User' in label:
                return 'User'
            if 'Gemini' in label:
                return 'Gemini'
        except Exception:
            return None
        return None

    def _parse_turns(self):
        """解析對話輪次 - 段落內再切分 User/Gemini，必要時呼叫 Gemini API"""
        # 使用長分隔線 (50+ 個 -) 切割對話段落
        segments = re.split(r'-{50,}', self.content)
        turns = []
        turn_id = 0

        # Gemini 回應的開頭標記(去除「謝謝你」避免誤判使用者致謝)
        gemini_starts = r'^(這是|這個|這份|這句話|這不|這種|聽到你|看著這|看著你|你剛|你說|你現在|你做|你的|你看|你會|試著|關於|當你|請|讓我|我想輕|我很想|我會|沒錯|哈哈)'

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # 用空行分段
            paragraphs = re.split(r'\n\s*\n', segment)

            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 15:
                    continue

                # 段落內可能包含 User+Gemini,需要再切分
                lines = para.split('\n')
                current_text = []
                current_speaker = None

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 判斷這行是否是新的 Gemini 回應開頭
                    is_gemini_start = bool(re.match(gemini_starts, line))
                    # 安全提示或列表亦視為 Gemini
                    if '專線' in line or re.search(r'[\d\*]\.\s+', line):
                        is_gemini_start = True
                    # 若以「謝謝你」開頭且包含「提醒我」則多為使用者致謝
                    if line.startswith('謝謝你') and '提醒我' in line:
                        is_gemini_start = False

                    # 如果檢測到 Gemini 開頭,且已有累積內容,先保存
                    if is_gemini_start and current_text and current_speaker == 'User':
                        text = '\n'.join(current_text)
                        turns.append({
                            'turn_id': turn_id,
                            'speaker': 'User',
                            'text': text,
                            'length': len(text)
                        })
                        turn_id += 1
                        current_text = []
                        current_speaker = 'Gemini'

                    # 第一行判斷發言者
                    if current_speaker is None:
                        if is_gemini_start:
                            current_speaker = 'Gemini'
                        else:
                            current_speaker = 'User'

                    current_text.append(line)

                # 保存段落最後的內容
                if current_text:
                    text = '\n'.join(current_text)
                    if len(text) > 15:
                        speaker = current_speaker
                        # 若開啟 Gemini API,嘗試覆寫判斷
                        api_label = self._classify_speaker_via_gemini(text) if self.use_gemini_api else None
                        if api_label in ('User', 'Gemini'):
                            speaker = api_label
                        turns.append({
                            'turn_id': turn_id,
                            'speaker': speaker,
                            'text': text,
                            'length': len(text)
                        })
                        turn_id += 1

        return turns
    
    def emotion_analysis(self):
        """情緒量化分析"""
        results = []
        
        for turn in self.turns:
            text = turn['text']
            emotion_scores = {}
            
            for emotion_type, keywords in self.emotion_dict.items():
                count = sum(text.count(kw) for kw in keywords)
                emotion_scores[emotion_type] = count
            
            results.append({
                'turn_id': turn['turn_id'],
                'speaker': turn['speaker'],
                **emotion_scores,
                'total_negative': emotion_scores['負面高強度'] + emotion_scores['負面中強度'],
                'total_positive': emotion_scores['正面低強度'] + emotion_scores['正面高強度']
            })
        
        return pd.DataFrame(results)
    
    def healing_strategy_analysis(self):
        """療癒策略分析"""
        strategy_usage = defaultdict(int)
        timeline = []
        
        for turn in self.turns:
            if turn['speaker'] == 'Gemini':
                text = turn['text']
                turn_strategies = {}
                
                for strategy, keywords in self.healing_strategies.items():
                    count = sum(text.count(kw) for kw in keywords)
                    strategy_usage[strategy] += count
                    turn_strategies[strategy] = count
                
                if any(turn_strategies.values()):
                    timeline.append({
                        'turn_id': turn['turn_id'],
                        **turn_strategies
                    })
        
        return dict(strategy_usage), pd.DataFrame(timeline)
    
    def risk_assessment(self):
        """自殺風險評估"""
        risk_timeline = []
        
        for turn in self.turns:
            if turn['speaker'] == 'User':
                text = turn['text']
                
                risk_count = sum(text.count(rf) for rf in self.risk_factors)
                protective_count = sum(text.count(pf) for pf in self.protective_factors)
                
                risk_timeline.append({
                    'turn_id': turn['turn_id'],
                    'risk_indicators': risk_count,
                    'protective_factors': protective_count,
                    'net_risk': risk_count - protective_count
                })
        
        return pd.DataFrame(risk_timeline)
    
    def thematic_analysis(self):
        """主題分析 (質性)"""
        themes = {
            '死亡與自殺意念': [],
            '孤獨與被遺棄': [],
            '創傷與失落': [],
            '自我照顧': [],
            '連結與支持': [],
            '存在與當下': []
        }
        
        theme_keywords = {
            '死亡與自殺意念': ['死', '自殺', '結束', '離開', '跳海'],
            '孤獨與被遺棄': ['一個人', '孤獨', '沒人愛', '寂寞'],
            '創傷與失落': ['PTSD', '大姊', '父親', '母親', '過世', '離世', '簽下'],
            '自我照顧': ['照顧自己', '疼自己', '煮麵', '吃麵', '休息'],
            '連結與支持': ['抱抱', '陪伴', '二姊', '三姊', '姪子', '姪女'],
            '存在與當下': ['呼吸', '存在', '當下', '看海', '海浪', '安靜']
        }
        
        for turn in self.turns:
            text = turn['text']
            for theme, keywords in theme_keywords.items():
                if any(kw in text for kw in keywords):
                    themes[theme].append({
                        'turn_id': turn['turn_id'],
                        'speaker': turn['speaker'],
                        'excerpt': text[:200] + '...' if len(text) > 200 else text
                    })
        
        return themes
    
    def narrative_arc_analysis(self):
        """敘事弧線分析 - 追蹤情緒轉變"""
        user_turns = [t for t in self.turns if t['speaker'] == 'User']
        
        narrative_points = []
        for turn in user_turns:
            text = turn['text'].lower()
            
            # 簡易情緒評分 (-5 to +5)
            score = 0
            score -= text.count('想死') * 3
            score -= text.count('痛苦') * 2
            score -= text.count('焦慮') * 1.5
            score -= text.count('無助') * 2
            score += text.count('平靜') * 2
            score += text.count('放鬆') * 1.5
            score += text.count('溫暖') * 1
            score += text.count('照顧自己') * 2
            
            narrative_points.append({
                'turn_id': turn['turn_id'],
                'emotional_valence': max(-5, min(5, score)),
                'key_phrase': text[:100]
            })
        
        return pd.DataFrame(narrative_points)
    
    def identify_turning_points(self):
        """識別關鍵轉折點 - 情緒/風險的重大變化"""
        narrative_df = self.narrative_arc_analysis()
        risk_df = self.risk_assessment()
        
        turning_points = []
        
        # 找出情緒變化超過2分的點
        for i in range(1, len(narrative_df)):
            prev_emotion = narrative_df.iloc[i-1]['emotional_valence']
            curr_emotion = narrative_df.iloc[i]['emotional_valence']
            change = curr_emotion - prev_emotion
            
            if abs(change) >= 2:
                turn_id = narrative_df.iloc[i]['turn_id']
                user_turn = next((t for t in self.turns if t['turn_id'] == turn_id), None)
                
                # 找前一個 Gemini 回應
                prev_gemini = None
                for t in reversed(self.turns[:turn_id]):
                    if t['speaker'] == 'Gemini':
                        prev_gemini = t
                        break
                
                turning_points.append({
                    'turn_id': turn_id,
                    'type': '情緒好轉' if change > 0 else '情緒惡化',
                    'change': change,
                    'user_text': user_turn['text'][:150] if user_turn else '',
                    'prev_gemini_strategy': self._extract_main_strategy(prev_gemini['text']) if prev_gemini else '無'
                })
        
        return pd.DataFrame(turning_points)
    
    def _extract_main_strategy(self, text):
        """從 Gemini 回應中提取主要策略"""
        strategy_counts = {}
        for strategy, keywords in self.healing_strategies.items():
            count = sum(text.count(kw) for kw in keywords)
            if count > 0:
                strategy_counts[strategy] = count
        
        if strategy_counts:
            return max(strategy_counts, key=strategy_counts.get)
        return '一般支持'
    
    def generate_report(self, output_dir='analysis_results'):
        """生成完整分析報告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 60)
        print("Gemini 自殺危機對話混合分析報告")
        print("=" * 60)
        
        # 1. 基本統計
        print("\n【1. 基本統計】")
        print(f"總對話輪次: {len(self.turns)}")
        user_turns = sum(1 for t in self.turns if t['speaker'] == 'User')
        print(f"使用者發言: {user_turns} 次")
        print(f"Gemini 回應: {len(self.turns) - user_turns} 次")
        
        # 2. 情緒分析
        print("\n【2. 情緒量化分析】")
        emotion_df = self.emotion_analysis()
        print(emotion_df[['speaker', '負面高強度', '正面低強度', 'total_negative', 'total_positive']].describe())
        emotion_df.to_csv(output_path / 'emotion_analysis.csv', index=False, encoding='utf-8-sig')
        
        # 3. 療癒策略
        print("\n【3. 療癒策略使用頻率】")
        strategies, strategy_timeline = self.healing_strategy_analysis()
        for strategy, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} 次")
        strategy_timeline.to_csv(output_path / 'healing_strategies.csv', index=False, encoding='utf-8-sig')
        
        # 4. 風險評估
        print("\n【4. 自殺風險時間軸】")
        risk_df = self.risk_assessment()
        print(f"平均風險指標: {risk_df['risk_indicators'].mean():.2f}")
        print(f"平均保護因子: {risk_df['protective_factors'].mean():.2f}")
        print(f"淨風險趨勢: {risk_df['net_risk'].mean():.2f}")
        risk_df.to_csv(output_path / 'risk_assessment.csv', index=False, encoding='utf-8-sig')
        
        # 5. 主題分析
        print("\n【5. 質性主題分析】")
        themes = self.thematic_analysis()
        for theme, excerpts in themes.items():
            print(f"\n{theme}: {len(excerpts)} 個片段")
            if excerpts:
                print(f"  範例: {excerpts[0]['excerpt'][:80]}...")
        
        with open(output_path / 'thematic_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(themes, f, ensure_ascii=False, indent=2)
        
        # 6. 敘事弧線
        print("\n【6. 情緒敘事弧線】")
        narrative_df = self.narrative_arc_analysis()
        print(narrative_df[['turn_id', 'emotional_valence']].to_string())
        narrative_df.to_csv(output_path / 'narrative_arc.csv', index=False, encoding='utf-8-sig')
        
        # 7. 轉折點分析
        print("\n【7. 關鍵轉折點】")
        turning_points = self.identify_turning_points()
        if not turning_points.empty:
            print(f"識別出 {len(turning_points)} 個重大轉折點:")
            for _, tp in turning_points.iterrows():
                print(f"  輪次 {tp['turn_id']}: {tp['type']} (變化{tp['change']:+.1f}) - 前次策略: {tp['prev_gemini_strategy']}")
            turning_points.to_csv(output_path / 'turning_points.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n\n[完成] 所有分析結果已儲存至: {output_path.absolute()}")
        
        return {
            'emotion_df': emotion_df,
            'strategies': strategies,
            'risk_df': risk_df,
            'themes': themes,
            'narrative_df': narrative_df,
            'turning_points': turning_points
        }
    
    def visualize(self, results, output_dir='analysis_results'):
        """視覺化分析結果"""
        output_path = Path(output_dir)
        
        # 1. 情緒時間軸
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        emotion_df = results['emotion_df']
        user_emotions = emotion_df[emotion_df['speaker'] == 'User']
        
        axes[0].plot(user_emotions['turn_id'], user_emotions['total_negative'], 
                     marker='o', label='負面情緒', color='red', linewidth=2)
        axes[0].plot(user_emotions['turn_id'], user_emotions['total_positive'], 
                     marker='s', label='正面情緒', color='green', linewidth=2)
        axes[0].set_xlabel('對話輪次')
        axes[0].set_ylabel('情緒強度')
        axes[0].set_title('使用者情緒變化時間軸')
        axes[0].set_xticks(range(0, int(user_emotions['turn_id'].max()) + 10, 10))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 風險評估時間軸
        risk_df = results['risk_df']
        axes[1].plot(risk_df['turn_id'], risk_df['net_risk'], 
                     marker='D', color='orange', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].fill_between(risk_df['turn_id'], risk_df['net_risk'], 0, 
                             where=(risk_df['net_risk'] > 0), alpha=0.3, color='red', label='高風險區')
        axes[1].fill_between(risk_df['turn_id'], risk_df['net_risk'], 0, 
                             where=(risk_df['net_risk'] <= 0), alpha=0.3, color='green', label='保護區')
        axes[1].set_xlabel('對話輪次')
        axes[1].set_ylabel('淨風險值')
        axes[1].set_title('自殺風險評估時間軸')
        axes[1].set_xticks(range(0, int(risk_df['turn_id'].max()) + 10, 10))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'emotion_risk_timeline.png', dpi=300, bbox_inches='tight')
        print(f"[完成] 圖表已儲存: {output_path / 'emotion_risk_timeline.png'}")
        
        # 3. 療癒策略長條圖
        strategies = results['strategies']
        if strategies:  # 只有在有策略資料時才繪製
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
            names, counts = zip(*sorted_strategies)
            
            colors = sns.color_palette("husl", len(names))
            ax.barh(names, counts, color=colors)
            ax.set_xlabel('使用次數')
            ax.set_title('Gemini 使用的療癒策略分佈')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'healing_strategies.png', dpi=300, bbox_inches='tight')
            print(f"[完成] 圖表已儲存: {output_path / 'healing_strategies.png'}")
        else:
            print(f"[警告] 無療癒策略資料,跳過該圖表")
        
        # 4. 敘事弧線
        fig, ax = plt.subplots(figsize=(12, 6))
        narrative_df = results['narrative_df']
        
        ax.plot(narrative_df['turn_id'], narrative_df['emotional_valence'], 
                marker='o', linewidth=2.5, markersize=8, color='purple')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.fill_between(narrative_df['turn_id'], narrative_df['emotional_valence'], 0,
                        where=(narrative_df['emotional_valence'] >= 0), alpha=0.3, color='blue', label='正向')
        ax.fill_between(narrative_df['turn_id'], narrative_df['emotional_valence'], 0,
                        where=(narrative_df['emotional_valence'] < 0), alpha=0.3, color='red', label='負向')
        ax.set_xlabel('對話輪次')
        ax.set_ylabel('情緒效價 (-5 to +5)')
        ax.set_title('情緒敘事弧線 (Narrative Arc)')
        ax.set_xticks(range(0, int(narrative_df['turn_id'].max()) + 10, 10))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'narrative_arc.png', dpi=300, bbox_inches='tight')
        print(f"[完成] 圖表已儲存: {output_path / 'narrative_arc.png'}")
        
        # 5. 主題分佈圓餅圖
        themes = results['themes']
        theme_counts = {theme: len(excerpts) for theme, excerpts in themes.items()}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("pastel", len(theme_counts))
        wedges, texts, autotexts = ax.pie(
            theme_counts.values(), 
            labels=theme_counts.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title('對話主題分佈', fontsize=16, pad=20)
        
        # 美化文字
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'theme_distribution.png', dpi=300, bbox_inches='tight')
        print(f"[完成] 圖表已儲存: {output_path / 'theme_distribution.png'}")
        
        # 6. 情緒熱力圖
        emotion_df = results['emotion_df']
        user_emotions = emotion_df[emotion_df['speaker'] == 'User'].copy()
        
        if len(user_emotions) > 0:
            # 準備熱力圖資料
            heatmap_data = user_emotions[['負面高強度', '負面中強度', '正面低強度', '正面高強度']].T
            
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.heatmap(heatmap_data, cmap='RdYlGn_r', cbar_kws={'label': '情緒強度'}, 
                       linewidths=0.5, ax=ax, vmin=0, vmax=10)
            ax.set_xticks(range(0, len(heatmap_data.columns), 10))
            ax.set_xticklabels(range(0, len(heatmap_data.columns), 10))
            ax.set_xlabel('對話輪次 (僅使用者發言)')
            ax.set_ylabel('情緒類型')
            ax.set_title('使用者情緒熱力圖')
            
            plt.tight_layout()
            plt.savefig(output_path / 'emotion_heatmap.png', dpi=300, bbox_inches='tight')
            print(f"[完成] 圖表已儲存: {output_path / 'emotion_heatmap.png'}")
        
        # 7. 詞雲圖 (User vs Gemini)
        if WordCloud is not None:
            # 中文停用詞列表
            stopwords = set([
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有', '看', '好', '自己', '這', '那', '這個', '那個', '可以', '但', '因為', '所以', '如果', '就是', '還是', '而且', '或者', '已經', '從', '讓', '把', '被', '給', '為', '與', '及', '以', '於', '對', '等', '之', '中', '又', '使', '對於', '只', '才', '能', '這樣', '那樣', '怎麼', '什麼', '誰', '哪', '些', '每', '可', '他', '她', '們', '其', '當', '便', '向', '將', '所', '比', '而', '此', '些', '這些', '那些', '嗎', '呢', '吧', '啊', '呀', '啦', '喔', '哦', '嘛', '吧', '嗎', '那麼', '這麼', '怎麼樣', '為什麼', '是不是', '還有', '並', '連', '不過', '然而', '但是', '即使', '雖然', '無論', '不管', '只要', '即便', '更', '最', '非常', '十分', '太', '真', '真的', '確實', '的確', '其實', '實際', '實在', '確定'
            ])
            
            user_text = ' '.join([t['text'] for t in self.turns if t['speaker'] == 'User'])
            gemini_text = ' '.join([t['text'] for t in self.turns if t['speaker'] == 'Gemini'])
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # User 詞雲
            if user_text:
                wc_user = WordCloud(font_path='C:/Windows/Fonts/msjh.ttc',
                                   width=800, height=400, 
                                   background_color='white',
                                   stopwords=stopwords,
                                   colormap='Reds').generate(user_text)
                axes[0].imshow(wc_user, interpolation='bilinear')
                axes[0].set_title('使用者高頻詞', fontsize=16)
                axes[0].axis('off')
            
            # Gemini 詞雲
            if gemini_text:
                wc_gemini = WordCloud(font_path='C:/Windows/Fonts/msjh.ttc',
                                     width=800, height=400,
                                     background_color='white',
                                     stopwords=stopwords,
                                     colormap='Blues').generate(gemini_text)
                axes[1].imshow(wc_gemini, interpolation='bilinear')
                axes[1].set_title('Gemini 高頻詞', fontsize=16)
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / 'wordcloud.png', dpi=300, bbox_inches='tight')
            print(f"[完成] 圖表已儲存: {output_path / 'wordcloud.png'}")
        
        # 8. 策略使用時間軸
        strategies = results['strategies']
        if strategies:
            strategy_timeline = []
            for turn in self.turns:
                if turn['speaker'] == 'Gemini':
                    text = turn['text']
                    turn_strategies = {}
                    for strategy, keywords in self.healing_strategies.items():
                        count = sum(text.count(kw) for kw in keywords)
                        if count > 0:
                            turn_strategies[strategy] = count
                    if turn_strategies:
                        strategy_timeline.append({
                            'turn_id': turn['turn_id'],
                            **turn_strategies
                        })
            
            if strategy_timeline:
                df_timeline = pd.DataFrame(strategy_timeline).fillna(0)
                df_timeline.set_index('turn_id', inplace=True)
                
                fig, ax = plt.subplots(figsize=(14, 6))
                df_timeline.plot(kind='area', stacked=True, ax=ax, 
                                colormap='tab10', alpha=0.7)
                ax.set_xlabel('對話輪次')
                ax.set_ylabel('策略使用次數')
                ax.set_title('療癒策略使用時間軸 (堆疊面積圖)')
                ax.set_xticks(range(0, int(df_timeline.index.max()) + 10, 10))
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'strategy_timeline.png', dpi=300, bbox_inches='tight')
                print(f"[完成] 圖表已儲存: {output_path / 'strategy_timeline.png'}")
        
        # 9. 風險-保護因子對比圖
        risk_df = results['risk_df']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = risk_df['turn_id']
        
        ax.bar(x, risk_df['risk_indicators'], label='風險指標', color='red', alpha=0.6)
        ax.bar(x, -risk_df['protective_factors'], label='保護因子', color='green', alpha=0.6)
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        ax.set_xlabel('對話輪次')
        ax.set_ylabel('指標數量')
        ax.set_title('風險指標 vs 保護因子對比')
        ax.set_xticks(range(0, int(risk_df['turn_id'].max()) + 10, 10))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path / 'risk_protection_compare.png', dpi=300, bbox_inches='tight')
        print(f"[完成] 圖表已儲存: {output_path / 'risk_protection_compare.png'}")
        
        # 10. 動態因果分析圖
        turning_points = results.get('turning_points')
        if turning_points is not None and not turning_points.empty:
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            # 10.1 情緒軌跡 + 轉折點標註
            emotion_df = results['emotion_df']
            user_emotions = emotion_df[emotion_df['speaker'] == 'User'].copy()
            
            # 繪製情緒強度總和
            user_emotions['total_intensity'] = (
                user_emotions['負面高強度'] + user_emotions['負面中強度'] - 
                user_emotions['正面低強度'] - user_emotions['正面高強度']
            )
            
            axes[0].plot(user_emotions['turn_id'], user_emotions['total_intensity'], 
                        marker='o', linewidth=2, color='steelblue', label='情緒強度')
            
            # 標註轉折點
            for idx, tp in turning_points.iterrows():
                tp_emotion = user_emotions[user_emotions['turn_id'] == tp['turn_id']]
                if not tp_emotion.empty:
                    y_val = tp_emotion['total_intensity'].values[0]
                    color = 'green' if tp['change'] < 0 else 'red'
                    axes[0].scatter(tp['turn_id'], y_val, s=200, c=color, 
                                   marker='*', edgecolors='black', linewidths=1.5, 
                                   zorder=10)
                    axes[0].annotate(f"{tp['prev_gemini_strategy']}", 
                                    xy=(tp['turn_id'], y_val),
                                    xytext=(0, 20), textcoords='offset points',
                                    ha='center', fontsize=8,
                                    bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.3),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[0].set_xlabel('對話輪次')
            axes[0].set_ylabel('淨情緒強度')
            axes[0].set_title('情緒軌跡與關鍵轉折點 (★=重大變化)')
            axes[0].set_xticks(range(0, int(user_emotions['turn_id'].max()) + 10, 10))
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 10.2 策略-效果關聯圖
            strategy_effect = turning_points.groupby('prev_gemini_strategy')['change'].agg(['mean', 'count'])
            strategy_effect = strategy_effect.sort_values('mean')
            
            colors = ['green' if x < 0 else 'red' for x in strategy_effect['mean']]
            axes[1].barh(strategy_effect.index, strategy_effect['mean'], color=colors, alpha=0.7)
            axes[1].axvline(x=0, color='black', linewidth=0.8)
            axes[1].set_xlabel('平均情緒變化 (負=改善, 正=惡化)')
            axes[1].set_ylabel('Gemini 使用的策略')
            axes[1].set_title('策略效果分析 (基於轉折點)')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # 在柱狀圖上標註使用次數
            for i, (idx, row) in enumerate(strategy_effect.iterrows()):
                axes[1].text(row['mean'], i, f" n={int(row['count'])}", 
                            va='center', fontsize=9, color='black')
            
            # 10.3 時間序列因果鏈
            axes[2].set_xlim(0, max(user_emotions['turn_id']) + 5)
            axes[2].set_ylim(-1, len(turning_points) + 1)
            
            for i, (idx, tp) in enumerate(turning_points.iterrows()):
                # 繪製事件節點
                y_pos = i
                color = 'green' if tp['change'] < 0 else 'red'
                
                # 標註事件點
                axes[2].scatter(tp['turn_id'], y_pos, s=300, c=color, 
                               marker='o', edgecolors='black', linewidths=2, zorder=5)
                
                # 事件文字
                axes[2].text(tp['turn_id'], y_pos + 0.3, 
                            f"轉折 #{tp['turn_id']}\n變化: {tp['change']:+.1f}", 
                            ha='center', fontsize=8, weight='bold')
                
                # 策略箭頭
                axes[2].annotate('', xy=(tp['turn_id'], y_pos), 
                                xytext=(tp['turn_id']-5, y_pos),
                                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
                axes[2].text(tp['turn_id']-5, y_pos - 0.3, 
                            f"策略: {tp['prev_gemini_strategy']}", 
                            ha='right', fontsize=7, style='italic',
                            bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.5))
            
            axes[2].set_xlabel('對話輪次')
            axes[2].set_xticks(range(0, int(user_emotions['turn_id'].max()) + 10, 10))
            axes[2].set_yticks([])
            axes[2].set_title('因果時間鏈: Gemini策略 → 使用者反應')
            axes[2].grid(True, alpha=0.2, axis='x')
            
            plt.tight_layout()
            plt.savefig(output_path / 'causal_analysis.png', dpi=300, bbox_inches='tight')
            print(f"[完成] 圖表已儲存: {output_path / 'causal_analysis.png'}")
            print("\n[視覺化完成] 共生成 10 組圖表")
        else:
            print("\n[視覺化完成] 共生成 9 組圖表 (無轉折點資料,跳過因果分析)")


def main():
    """主程式"""
    # 設定檔案路徑
    file_path = r'c:\Users\laiweicheng\Downloads\新增資料夾 (5)\Gemini_Suicide.txt'
    
    # 設定輸出路徑到新增資料夾(5)內的 analysis_results
    output_dir = r'c:\Users\laiweicheng\Downloads\新增資料夾 (5)\analysis_results'

    # 若環境變數有 GEMINI_API_KEY 則啟用 Gemini API 判斷發話者
    use_gemini_api = bool(os.environ.get('GEMINI_API_KEY'))
    
    print("開始分析 Gemini 對話...")
    print(f"檔案: {file_path}")
    print(f"輸出路徑: {output_dir}\n")
    print(f"發話者判斷: {'Gemini API' if use_gemini_api else '規則判斷'}\n")
    
    # 建立分析器
    analyzer = GeminiConversationAnalyzer(file_path, use_gemini_api=use_gemini_api)
    
    # 生成報告 - 指定輸出路徑
    results = analyzer.generate_report(output_dir=output_dir)
    
    # 視覺化 - 指定輸出路徑
    print("\n生成視覺化圖表...")
    analyzer.visualize(results, output_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
