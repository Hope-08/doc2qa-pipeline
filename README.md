# doc2qa-pipeline

企业知识库 QA 数据自动生成平台

## 快速开始

### 1. 环境要求

- Python 3.11+
- uv 包管理器
- **Tesseract OCR**（扫描 PDF / OCR 功能，[Windows 下载](https://github.com/UB-Mannheim/tesseract/wiki)）
- **Ghostscript**（表格提取 / ocrmypdf 依赖，[下载地址](https://www.ghostscript.com/releases/)）

### 2. 安装

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 克隆项目（或直接在现有目录）
cd doc2qa-pipeline

# 同步依赖（自动创建虚拟环境）
uv sync

# 安装文档解析依赖（本地多引擎）
uv sync --extra parsing

# 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows
```

### 3. 快速测试

```bash
# 使用 uv run（推荐，自动管理环境）
uv run python main.py run \
    --input sample.pdf

# 指定输出路径（默认写入 output/{job_id}/qa_dataset.jsonl）
uv run python main.py run \
    --input sample.pdf \
    --output my_dataset.jsonl

# 或在激活的虚拟环境中
python main.py run --input sample.pdf
```

运行完毕后，中间结果保存在 `output/{job_id}/` 目录：

```
output/
  20260415_143022_7381/
    01_parsed_sample.json     # 解析原始结构
    02_normalized_sample.json # 清洗后内容
    03_chunks_sample.json     # 分块列表
    04_qa_raw_sample.json     # 生成的原始 QA
    05_qa_filtered.json       # 质检+去重后 QA（含评分）
    qa_dataset.jsonl          # 最终导出
```

### 4. 配置

复制环境变量模板：

```bash
cp .env.example .env
# 编辑 .env，填入 API Keys 和解析配置
```

**关键配置项**：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PARSING_BACKEND` | `auto` | 解析引擎：auto \| docling \| pymupdf \| unstructured |
| `OCR_LANG` | `chi_sim+eng` | Tesseract 语言包 |
| `LLM_MODEL` | `azure-openai/gpt-5.4` | LLM 模型名称，如 `gpt-4o`、`qwen2.5-72b-instruct` |
| `QA_STRATEGY` | `simple` | QA 生成策略：`simple`（逐块）\| `advanced`（文档级多跳） |
| `MAX_QA_PER_DOC` | `50` | Advanced 策略下单文档最多生成 QA 数 |

> 无需配置外部 API 服务，文档解析完全在本机运行。

## 核心功能

- ✅ 多格式文档解析（PDF/DOCX/图片）
- ✅ 扫描PDF自动OCR（Tesseract，多语言）
- ✅ 表格结构保留（HTML格式）
- ✅ heading 边界感知分块（HeadlineSplitter，章节级别切分）
- ✅ 语义感知分块（基于embedding，短文本自动预合并）
- ✅ 多类型QA生成（definition/reasoning/comparison等）
- ✅ Advanced QA生成（实体提取 + 跨块关系图 + 多跳推理问题）
- ✅ 自一致性过滤（淘汰无需上下文即可回答的过简问题）
- ✅ 低质量chunk自动过滤（过短、纯日期列表等）
- ✅ NLI幻觉检测（faithfulness评分）
- ✅ 断点续跑（Prefect编排）
- ✅ 本地多引擎解析（PyMuPDF/docling/ocrmypdf，无需 GPU）
- ✅ RESTful API 服务（文件上传 → 异步任务 → 结果查询）
- ✅ 每步中间结果持久化（`output/{job_id}/`，便于排查质量问题）

## 文档

- [实现方案](./实现方案.md) - 完整技术方案（12章节）
- [初步方案](./初步方案.md) - 原始方案
- API文档 - *待补充*

## 依赖管理

本项目使用 **uv** 管理依赖，配置在 `pyproject.toml` 中。

```bash
# 添加新依赖
uv add "package-name>=1.0.0"

# 添加开发依赖
uv add --dev "pytest>=7.0.0"

# 升级依赖
uv sync --upgrade

# 安装可选依赖组
uv sync --extra dev      # 开发工具
uv sync --extra parsing  # 文档解析引擎
uv sync --extra api      # API服务
```

## 开发

### 运行测试

```bash
uv run pytest
```

### 代码格式化

```bash
uv run black .
uv run ruff check --fix .
```

### 类型检查

```bash
uv run mypy .
```

## 部署

### Docker

```bash
# 构建
docker build -t doc2qa-pipeline:latest .

# 运行
docker run -v $(pwd)/docs:/app/docs \
           -v $(pwd)/output:/app/output \
           doc2qa-pipeline:latest \
           run \
           --input /app/docs \
           --output /app/output/dataset.jsonl
```

## 技术栈

| 组件 | 技术选型 | 版本 |
|------|---------|------|
| 文档解析（主力） | docling | 2.0.0+ |
| 文档解析（快速） | PyMuPDF | 1.24.0+ |
| 扫描OCR | ocrmypdf + Tesseract | 16.0.0+ |
| 表格提取 | camelot-py + img2table | 0.11.0+ |
| 语义分块 | langchain-openai Embeddings | - |
| QA生成（简单） | langchain-openai | 0.1.0+ |
| QA生成（多跳） | 实体提取 + 关系图 + 合成（LLM） | - |
| 质量评估 | LLM NLI + OpenAI Embeddings | - |
| Pipeline编排 | Prefect | 3.0.0+ |
| API服务 | FastAPI + uvicorn | 0.100.0+ |

## 许可证

*待定*

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系

*待补充*
