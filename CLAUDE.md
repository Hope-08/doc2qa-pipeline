# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 安装依赖（核心依赖）
uv sync

# 安装文档解析依赖（本地多引擎）
uv sync --extra parsing

# 安装开发依赖（含 pytest、ruff、black、mypy）
uv sync --extra dev

# 安装 API 服务依赖
uv sync --extra api

# 安装所有可选依赖
uv sync --all-extras

# 运行所有单元测试
uv run pytest tests/unit/

# 运行单个测试文件
uv run pytest tests/unit/test_document_parser.py -v

# 运行单个测试函数
uv run pytest tests/unit/test_document_parser.py::TestDocumentParser::test_parse_success -v

# Lint + 格式检查
uv run ruff check .
uv run black --check .

# 自动修复
uv run ruff check --fix .
uv run black .

# 运行 Pipeline CLI（输出自动写入 output/{job_id}/ 目录）
uv run python main.py run --input docs/
uv run python main.py run --input docs/ --model gpt-4o --format alpaca
uv run python main.py run --input docs/ --output my_dataset.jsonl  # 覆盖默认输出路径
uv run python main.py --debug run --input sample.pdf  # DEBUG 日志

# 启动 RESTful API 服务
uv run python serve.py --reload    # 开发模式，http://localhost:8080/docs
uv run python serve.py             # 生产模式
```

## 架构概览

### 数据流

```
文档文件
  → parse_task        → ParsedDocument           → 01_parsed_{doc}.json
  → normalize_task    → ParsedDocument (cleaned) → 02_normalized_{doc}.json
  → chunk_task        → list[SemanticChunk]       → 03_chunks_{doc}.json
  ┌─ simple 策略 ────────────────────────────────────────────────────────────
  │ → generate_qa_task  → list[QAPair]            → 04_qa_raw_{doc}.json
  └─ advanced 策略 ──────────────────────────────────────────────────────────
    → enrich_task      → list[EnrichedChunk]      → 04_enriched_{doc}.json
    → synthesize_qa_task → list[QAPair]           → 04_qa_raw_{doc}.json
  → score_filter_task → list[QAPair] (质检+去重)  → 05_qa_filtered.json
  → export_task       → JSONL / HF Dataset / Alpaca / RAGAS
                                                  → qa_dataset.{ext}
```

所有中间结果写入 `output/{job_id}/`（CLI 用时间戳+随机数，API 用 job_id UUID）。  
每个阶段对应 `pipeline/tasks.py` 中的一个 Prefect `@task`。Flow 编排在 `pipeline/flow.py`。

### 中间结果目录（`output/{job_id}/`）

每次运行在 `output/` 下创建一个独立子目录，保存每步结果，便于排查质量问题：

```
output/
  20260415_143022_7381/        ← CLI 运行（时间戳_随机数）
  a3f2c1d4-xxxx-xxxx/          ← API 运行（job_id UUID）
    01_parsed_{doc}.json       # 解析原始结果（content_list + markdown）
    02_normalized_{doc}.json   # 清洗后（过滤噪声 ContentItem）
    03_chunks_{doc}.json       # 分块列表（内容、类型、page_idx）
    04_enriched_{doc}.json     # Advanced 策略：实体富化后的 EnrichedChunk 列表
    04_qa_raw_{doc}.json       # 未过滤 QA（可对比过滤前后差异）
    04_qa_raw_all.json         # 多文档时的汇总（单文档时无此文件）
    05_qa_filtered.json        # 质检+去重后 QA（含所有评分字段）
    qa_dataset.jsonl           # 最终导出文件
```

排查步骤：
1. `03_chunks` → 确认分块是否合理（数量、长度、类型分布）
2. `04_enriched` → Advanced 策略下确认实体提取结果（entities、summary）
3. `04_qa_raw` → 确认 LLM 生成了什么（过滤前原始问答）
4. `05_qa_filtered` → 查看 `is_passed=false` 的条目和 `fail_reasons`

### 核心数据模型（`models.py`）

所有模块共享的 dataclass，**集中定义在根目录的 `models.py`** 以避免循环导入：

| 类 | 用途 |
|----|------|
| `ContentItem` | 解析器输出的单个内容项（text / table / image / equation） |
| `ParsedDocument` | 单个文档的解析结果，含 `content_list` 和 `markdown` |
| `SemanticChunk` | 分块后的语义单元，`content_type` 为 text / table / image_caption |
| `EnrichedChunk` | Advanced QA 用的富化分块，含 `entities`（关键实体列表）和 `summary` |
| `QAPair` | 问答对，`quality_score` 等字段由 qa_filtering 填充 |
| `QualityThresholds` | 质量过滤阈值，frozen dataclass |

### 配置（`config/settings.py`）

基于 `pydantic-settings`，优先级：**环境变量 > `.env` 文件 > 字段默认值**。

关键配置项：

| 字段 | 环境变量 | 说明 |
|------|---------|------|
| `parsing_backend` | `PARSING_BACKEND` | 解析引擎：auto \| docling \| pymupdf \| unstructured |
| `ocr_lang` | `OCR_LANG` | Tesseract 语言包，如 `chi_sim+eng` |
| `llm_model` | `LLM_MODEL` | QA 生成模型，默认 `azure-openai/gpt-5.4`，可改为 `gpt-4o` |
| `embedding_model` | `EMBEDDING_MODEL` | OpenAI embedding 模型，默认 `azure-openai/text-embedding-3-small` |
| `nli_model` | `NLI_MODEL` | faithfulness 评分 LLM，默认 `azure-openai/gpt-5.4` |
| `qa_strategy` | `QA_STRATEGY` | QA 生成策略：`simple`（逐块）\| `advanced`（文档级多跳） |
| `max_qa_per_doc` | `MAX_QA_PER_DOC` | Advanced 策略下单文档最多生成 QA 数，默认 50 |
| `self_consistency_check` | `SELF_CONSISTENCY_CHECK` | 自一致性过滤开关，默认 true |
| `entity_extractor_model` | `ENTITY_EXTRACTOR_MODEL` | 实体提取 LLM，默认同 `nli_model` |
| `synthesize_model` | `SYNTHESIZE_MODEL` | 多跳 QA 合成 LLM，默认同 `llm_model` |

模块内通过 `from config.settings import settings` 获取全局单例（懒加载 `.env`）。

### 本地多引擎解析（`parsing/document_parser.py`）

全部在本机运行，无需外部服务，无需 GPU。解析策略（`backend=auto`）：

- **PDF 数字版**（平均 >100 字符/页）→ PyMuPDF 直接提取（毫秒级）
- **PDF 扫描版** → ocrmypdf 生成文字层 → docling 精解析
- **DOCX / 图片** → docling（内置 OCR）
- **兜底** → unstructured

输出格式（`ParsedDocument` / `ContentItem`）与原版完全一致，下游模块零改动。

**系统依赖**（需手动安装）：
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)（Windows 安装包）
- [Ghostscript](https://www.ghostscript.com/releases/)（camelot / ocrmypdf 依赖）

### Prefect Task 序列化（`pipeline/tasks.py`）

Prefect 要求任务返回值可 JSON 化，因此每个 Task 的返回值均为 `dict` / `list[dict]`（通过 `_serialize()` 将 dataclass 转换）。下游 Task 收到 dict 后需手动反序列化：

```python
items = [ContentItem(**item) for item in parsed_dict["content_list"]]
```

### 分块策略（`chunking/`）

- `TypeAwareChunker`（入口）：按文档 heading 数量决定分块路径
  - heading 数 ≥ `headline_min_count`（默认 3） → **HeadlineSplitter**：按 heading 边界分章节，章节过长时二次调用 `SemanticChunker`；超长章节切出的子 chunk 通过字符偏移精确分配 `source_items`
  - heading 不足 → 旧逻辑：text → 短文本预合并 → SemanticChunker，table/image → 整体保留
- 预合并阈值 `_MERGE_THRESHOLD = 200`：避免每个标题/行程条目单独成 chunk 导致碎片化
- 语义切分 embedding API 按**批次**调用（单次请求），减少 API 调用次数

### QA 生成（`qa_generation/`）

**Simple 策略**（默认，`QA_STRATEGY=simple`）：

`QAGenerator.generate_batch()` 在调用 LLM 前先过滤低质量 chunk：

- **过短**（< 50 字符）→ 跳过，没有足够上下文
- **纯日期列表**（日期格式字符占比 > 60%）→ 跳过，无领域知识

Prompt 模板（`config/prompts.py`）对 `text` 类型明确禁止：元文本分析题、同义反复题、不完整信息题、纯数据罗列题。若整段文本无领域知识，要求 LLM 返回 `[]`。

**Advanced 策略**（`QA_STRATEGY=advanced`）：

```
SemanticChunk[]
  → enrich_task（entity_extractor.py）
      为每个 chunk 调用 LLM 提取关键实体 → list[EnrichedChunk]
  → synthesize_qa_task（doc_qa_synthesizer.py）
      1. doc_relation_index.py：按实体重叠 + embedding 相似度建跨块关系图
      2. advanced_generator.py：基于相关 chunk 对生成多跳推理 QA
      3. 按文档字符数分配 quota（max_qa_per_doc 上限）
      4. 自一致性验证（self_consistency_check）：问题能被 LLM 直接回答 → 丢弃
  → list[QAPair]（含 cross-chunk 多跳问题）
```

`pipeline/flow.py` 按 `cfg.qa_strategy` 路由；若 `enrich_task` 返回空列表则跳过该文档并写警告日志。

### 质量过滤（`qa_filtering/`）

评分维度权重：Faithfulness 50%、Relevance 30%、Completeness 20%。  
- **faithfulness**：调用 LLM（`nli_model`）批量判断 answer 是否由 context 支撑，返回 0~1 分数  
- **relevance**：OpenAI Embeddings 计算 Q-A 余弦相似度  
- **completeness**：规则评分（答案长度 + 是否含句号）  
`QualityScorer.score_batch()` 返回新 QAPair 列表（不修改原对象）。

### 测试模式

单元测试使用 `unittest.mock` 模拟外部依赖：

- **DocumentParser 测试**：Mock 各本地引擎
  - `patch("fitz.open")` → 模拟 PyMuPDF 结果
  - `patch("docling.document_converter.DocumentConverter")` → 模拟 docling 结果
  - `patch("unstructured.partition.auto.partition")` → 模拟 unstructured 结果
- **LLM 相关测试**：Mock `langchain_openai.ChatOpenAI` 和 `langchain_openai.OpenAIEmbeddings`
- **Advanced QA 测试**：Mock 各子模块
  - `patch("qa_generation.entity_extractor.EntityExtractor")` → 模拟实体提取
  - `patch("qa_generation.doc_qa_synthesizer.calc_doc_quota")` → 注意：`flow.py` 懒导入，patch 路径为源模块
  - `patch("qa_generation.multi_type_generator.QAGenerator")` → 模拟 simple 路径的生成器
- **Flow 路由测试**：使用 `.fn()` 绕过 Prefect wrapper，见 `tests/unit/test_flow_routing.py`
- **共享 Fixture**：见 `tests/conftest.py`

### 可选依赖组

| 组 | 内容 |
|----|------|
| `qa` | ragas、deepeval、langchain-core |
| `api` | fastapi、uvicorn、python-multipart |
| `parsing` | docling、PyMuPDF、unstructured、ocrmypdf、camelot-py、img2table（本地多引擎解析） |
| `dev` | pytest、ruff、black、mypy |

`uv sync`（不加 extra）只安装核心依赖（langchain-openai、prefect 等），embedding 和 LLM 均通过 OpenAI API 调用，无需本地 ML 栈。
