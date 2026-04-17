"""文本过滤规则：供 qa_generation 和 qa_filtering 共用的正则/常量。"""

from __future__ import annotations

import re

# 元文本指代词正则：问题或答案中含有这些词则判定为元文本问题。
# 同时被 multi_type_generator（生成时过滤）和 quality_scorer（评分时惩罚）使用。
META_TEXT_RE = re.compile(
    r"(这段文[本字]|这段内容|这篇文[章章]|文中[提到说描述]|根据(本文|材料|文本|上文|文章)|"
    r"文章[指出提到说明]|从文[章中]可以|根据以上|上文所述|如文中所[示说述]|"
    r"文章所说|文本[表明指出说明]|这篇[文文章]|原文[中提到说明]|"
    r"from (the )?(text|passage|document|article)|according to (the )?(text|passage|document|article)|"
    r"as (mentioned|stated|described) in (the )?(text|passage|document))",
    re.IGNORECASE,
)
