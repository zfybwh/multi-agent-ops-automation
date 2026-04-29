"""
Multi-Agent 协同运营自动化系统 - 单文件可运行 MVP

功能：
1. 接收运营目标，例如：提升新用户 7 日留存、召回沉默用户、提高活动转化。
2. Planner Agent 将目标拆解成运营任务。
3. Data Analyst Agent 读取用户与事件数据，生成用户分群和洞察。
4. Content Agent 生成不同分群的触达文案。
5. Campaign Agent 生成触达策略和执行计划。
6. QA Agent 做风险、合规、重复触达检查。
7. Executor Agent 将通过审核的任务写入执行队列。
8. FastAPI 提供接口，SQLite 持久化数据。

运行方式：
    pip install fastapi uvicorn pydantic python-dotenv openai
    uvicorn multi_agent_ops_automation_system:app --reload --port 8000

可选：
    export OPENAI_API_KEY="你的 key"

测试：
    curl -X POST http://127.0.0.1:8000/campaigns/run \
      -H "Content-Type: application/json" \
      -d '{"goal":"召回 30 天未活跃但历史消费超过 100 元的用户","channel":"push","budget":5000}'
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# ============================================================
# 1. 基础配置
# ============================================================

DB_PATH = os.getenv("OPS_DB_PATH", "ops_automation.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class Channel(str, Enum):
    push = "push"
    email = "email"
    sms = "sms"
    in_app = "in_app"


class TaskStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    executed = "executed"


class CampaignRequest(BaseModel):
    goal: str = Field(..., description="运营目标")
    channel: Channel = Field(default=Channel.push, description="触达渠道")
    budget: float = Field(default=0, description="预算")
    operator: str = Field(default="ops_user", description="运营负责人")


class CampaignResponse(BaseModel):
    campaign_id: str
    goal: str
    summary: Dict[str, Any]
    segments: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    qa_result: Dict[str, Any]


# ============================================================
# 2. 数据库
# ============================================================

class Database:
    def __init__(self, path: str):
        self.path = path
        self.init_db()

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    last_active_at TEXT,
                    total_spend REAL,
                    lifecycle_stage TEXT,
                    tags TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    event_type TEXT,
                    event_time TEXT,
                    properties TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    goal TEXT,
                    channel TEXT,
                    budget REAL,
                    operator TEXT,
                    created_at TEXT,
                    plan_json TEXT,
                    segments_json TEXT,
                    qa_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    segment_name TEXT,
                    channel TEXT,
                    message_title TEXT,
                    message_body TEXT,
                    schedule_time TEXT,
                    status TEXT,
                    reason TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()

    def seed_demo_data(self):
        with self.connect() as conn:
            existing = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
            if existing > 0:
                return

            now = datetime.utcnow()
            demo_users = [
                ("u001", "Alice", "alice@example.com", "13800000001", now - timedelta(days=3), 20, "new", ["app_user"]),
                ("u002", "Bob", "bob@example.com", "13800000002", now - timedelta(days=35), 188, "mature", ["paid", "silent"]),
                ("u003", "Cindy", "cindy@example.com", "13800000003", now - timedelta(days=40), 320, "vip", ["paid", "silent", "high_value"]),
                ("u004", "David", "david@example.com", "13800000004", now - timedelta(days=8), 0, "new", ["trial"]),
                ("u005", "Eva", "eva@example.com", "13800000005", now - timedelta(days=60), 30, "churn", ["silent"]),
                ("u006", "Frank", "frank@example.com", "13800000006", now - timedelta(days=1), 99, "active", ["active"]),
            ]
            for user in demo_users:
                conn.execute(
                    """
                    INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user[0],
                        user[1],
                        user[2],
                        user[3],
                        user[4].isoformat(),
                        user[5],
                        user[6],
                        json.dumps(user[7], ensure_ascii=False),
                    ),
                )
            conn.commit()

    def fetch_users(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM users").fetchall()
            result = []
            for r in rows:
                item = dict(r)
                item["tags"] = json.loads(item.get("tags") or "[]")
                result.append(item)
            return result

    def save_campaign(
        self,
        campaign_id: str,
        req: CampaignRequest,
        plan: Dict[str, Any],
        segments: List[Dict[str, Any]],
        qa_result: Dict[str, Any],
    ):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO campaigns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    req.goal,
                    req.channel.value,
                    req.budget,
                    req.operator,
                    datetime.utcnow().isoformat(),
                    json.dumps(plan, ensure_ascii=False),
                    json.dumps(segments, ensure_ascii=False),
                    json.dumps(qa_result, ensure_ascii=False),
                ),
            )
            conn.commit()

    def save_tasks(self, campaign_id: str, tasks: List[Dict[str, Any]]):
        with self.connect() as conn:
            for task in tasks:
                conn.execute(
                    """
                    INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task["task_id"],
                        campaign_id,
                        task["segment_name"],
                        task["channel"],
                        task["message_title"],
                        task["message_body"],
                        task["schedule_time"],
                        task["status"],
                        task.get("reason", ""),
                        datetime.utcnow().isoformat(),
                    ),
                )
            conn.commit()

    def list_campaigns(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM campaigns ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def list_tasks(self, campaign_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            if campaign_id:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE campaign_id = ? ORDER BY created_at DESC",
                    (campaign_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]


# ============================================================
# 3. LLM 网关：有 Key 时调用模型，无 Key 时走本地规则兜底
# ============================================================

class LLMGateway:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key and OpenAI else None

    def complete_json(self, system_prompt: str, user_prompt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return fallback

        try:
            resp = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            text = resp.choices[0].message.content or "{}"
            return json.loads(text)
        except Exception:
            return fallback


# ============================================================
# 4. Agent 基类和上下文
# ============================================================

@dataclass
class AgentContext:
    request: CampaignRequest
    campaign_id: str
    users: List[Dict[str, Any]]
    plan: Optional[Dict[str, Any]] = None
    insights: Optional[Dict[str, Any]] = None
    segments: Optional[List[Dict[str, Any]]] = None
    contents: Optional[List[Dict[str, Any]]] = None
    tasks: Optional[List[Dict[str, Any]]] = None
    qa_result: Optional[Dict[str, Any]] = None


class BaseAgent:
    name = "base_agent"

    def __init__(self, llm: LLMGateway):
        self.llm = llm

    def run(self, ctx: AgentContext) -> AgentContext:
        raise NotImplementedError


# ============================================================
# 5. 多 Agent 实现
# ============================================================

class PlannerAgent(BaseAgent):
    name = "planner_agent"

    def run(self, ctx: AgentContext) -> AgentContext:
        fallback = {
            "objective": ctx.request.goal,
            "strategy": [
                "识别目标用户分群",
                "分析分群痛点和转化阻力",
                "为不同分群生成差异化文案",
                "设置触达时间和频控规则",
                "审核风险后进入执行队列",
            ],
            "success_metrics": ["触达率", "点击率", "转化率", "退订率", "7 日留存"],
            "constraints": ["避免频繁打扰用户", "不得承诺无法兑现的权益", "敏感人群需降级触达"],
        }
        plan = self.llm.complete_json(
            system_prompt="你是资深增长运营负责人。请只输出 JSON。",
            user_prompt=f"请为运营目标生成自动化运营计划：{ctx.request.goal}，渠道：{ctx.request.channel}，预算：{ctx.request.budget}",
            fallback=fallback,
        )
        ctx.plan = plan
        return ctx


class DataAnalystAgent(BaseAgent):
    name = "data_analyst_agent"

    def run(self, ctx: AgentContext) -> AgentContext:
        now = datetime.utcnow()
        silent_high_value = []
        silent_low_value = []
        new_users = []
        active_users = []

        for u in ctx.users:
            last_active = datetime.fromisoformat(u["last_active_at"])
            inactive_days = (now - last_active).days
            spend = float(u["total_spend"])

            enriched = {**u, "inactive_days": inactive_days}
            if inactive_days >= 30 and spend >= 100:
                silent_high_value.append(enriched)
            elif inactive_days >= 30:
                silent_low_value.append(enriched)
            elif u["lifecycle_stage"] == "new" or inactive_days <= 7:
                new_users.append(enriched)
            else:
                active_users.append(enriched)

        segments = [
            {
                "name": "高价值沉默用户",
                "description": "30 天以上未活跃，历史消费金额较高，适合权益召回。",
                "user_count": len(silent_high_value),
                "user_ids": [u["user_id"] for u in silent_high_value],
                "recommended_offer": "专属回归券或会员权益",
            },
            {
                "name": "低价值沉默用户",
                "description": "30 天以上未活跃，历史消费较低，适合轻量内容唤醒。",
                "user_count": len(silent_low_value),
                "user_ids": [u["user_id"] for u in silent_low_value],
                "recommended_offer": "内容推荐或限时提醒",
            },
            {
                "name": "新用户激活用户",
                "description": "近期注册或近期活跃但消费较低，适合引导完成关键行为。",
                "user_count": len(new_users),
                "user_ids": [u["user_id"] for u in new_users],
                "recommended_offer": "新手任务奖励",
            },
            {
                "name": "稳定活跃用户",
                "description": "近期活跃用户，适合交叉销售或活动提醒。",
                "user_count": len(active_users),
                "user_ids": [u["user_id"] for u in active_users],
                "recommended_offer": "活动提醒或进阶权益",
            },
        ]
        segments = [s for s in segments if s["user_count"] > 0]
        ctx.segments = segments
        ctx.insights = {
            "total_users": len(ctx.users),
            "segment_count": len(segments),
            "key_finding": "沉默高价值用户是优先级最高的召回对象；新用户需要引导完成关键行为。",
        }
        return ctx


class ContentAgent(BaseAgent):
    name = "content_agent"

    def run(self, ctx: AgentContext) -> AgentContext:
        contents = []
        for segment in ctx.segments or []:
            fallback = self._fallback_content(ctx, segment)
            result = self.llm.complete_json(
                system_prompt="你是增长运营文案专家。请只输出 JSON，字段为 title、body、cta、risk_note。",
                user_prompt=json.dumps(
                    {
                        "goal": ctx.request.goal,
                        "channel": ctx.request.channel.value,
                        "segment": segment,
                        "plan": ctx.plan,
                    },
                    ensure_ascii=False,
                ),
                fallback=fallback,
            )
            contents.append({"segment_name": segment["name"], **result})
        ctx.contents = contents
        return ctx

    def _fallback_content(self, ctx: AgentContext, segment: Dict[str, Any]) -> Dict[str, Any]:
        name = segment["name"]
        offer = segment.get("recommended_offer", "专属权益")
        if "高价值沉默" in name:
            return {
                "title": "你的专属回归权益已到账",
                "body": f"好久不见，我们为你准备了{offer}。现在回来看看，限时可用。",
                "cta": "立即领取",
                "risk_note": "避免夸大权益，确保优惠真实可用。",
            }
        if "低价值沉默" in name:
            return {
                "title": "有新内容值得看看",
                "body": "我们根据你的兴趣更新了一批内容，回来看看有没有你喜欢的。",
                "cta": "去看看",
                "risk_note": "减少营销压力，控制触达频率。",
            }
        if "新用户" in name:
            return {
                "title": "完成新手任务，解锁奖励",
                "body": "只差一步就能完成首次体验，跟着指引操作即可获得奖励。",
                "cta": "继续完成",
                "risk_note": "文案需要清晰说明奖励规则。",
            }
        return {
            "title": "本周活动正在进行",
            "body": "你关注的权益和活动已经更新，点击查看详情。",
            "cta": "查看活动",
            "risk_note": "避免对活跃用户过度打扰。",
        }


class CampaignAgent(BaseAgent):
    name = "campaign_agent"

    def run(self, ctx: AgentContext) -> AgentContext:
        tasks = []
        start = datetime.utcnow() + timedelta(minutes=10)
        for index, content in enumerate(ctx.contents or []):
            tasks.append(
                {
                    "task_id": str(uuid.uuid4()),
                    "segment_name": content["segment_name"],
                    "channel": ctx.request.channel.value,
                    "message_title": content["title"],
                    "message_body": content["body"] + f" CTA：{content.get('cta', '查看详情')}",
                    "schedule_time": (start + timedelta(hours=index * 2)).isoformat(),
                    "status": TaskStatus.pending.value,
                    "reason": "等待 QA Agent 审核",
                }
            )
        ctx.tasks = tasks
        return ctx


class QAAgent(BaseAgent):
    name = "qa_agent"

    FORBIDDEN_WORDS = ["稳赚", "绝对", "永久免费", "内部特权", "最后一次机会"]

    def run(self, ctx: AgentContext) -> AgentContext:
        approved_tasks = []
        rejected_tasks = []
        warnings = []

        for task in ctx.tasks or []:
            text = task["message_title"] + " " + task["message_body"]
            hit_words = [w for w in self.FORBIDDEN_WORDS if w in text]
            too_long = len(task["message_body"]) > 180 and task["channel"] in [Channel.push.value, Channel.sms.value]

            if hit_words:
                task["status"] = TaskStatus.rejected.value
                task["reason"] = f"命中高风险词：{','.join(hit_words)}"
                rejected_tasks.append(task)
                warnings.append(task["reason"])
            elif too_long:
                task["status"] = TaskStatus.rejected.value
                task["reason"] = "短渠道文案过长，可能影响用户体验"
                rejected_tasks.append(task)
                warnings.append(task["reason"])
            else:
                task["status"] = TaskStatus.approved.value
                task["reason"] = "审核通过"
                approved_tasks.append(task)

        ctx.tasks = approved_tasks + rejected_tasks
        ctx.qa_result = {
            "approved_count": len(approved_tasks),
            "rejected_count": len(rejected_tasks),
            "warnings": warnings,
            "rules": ["敏感词检查", "短渠道长度检查", "频控建议", "权益真实性检查"],
        }
        return ctx


class ExecutorAgent(BaseAgent):
    name = "executor_agent"

    def __init__(self, llm: LLMGateway, db: Database):
        super().__init__(llm)
        self.db = db

    def run(self, ctx: AgentContext) -> AgentContext:
        executable = []
        for task in ctx.tasks or []:
            if task["status"] == TaskStatus.approved.value:
                task["status"] = TaskStatus.executed.value
                task["reason"] = "已写入执行队列；真实生产环境可在此调用短信、邮件、Push 或营销云 API"
            executable.append(task)

        self.db.save_campaign(
            campaign_id=ctx.campaign_id,
            req=ctx.request,
            plan=ctx.plan or {},
            segments=ctx.segments or [],
            qa_result=ctx.qa_result or {},
        )
        self.db.save_tasks(ctx.campaign_id, executable)
        ctx.tasks = executable
        return ctx


# ============================================================
# 6. 编排器：Agent 协同核心逻辑
# ============================================================

class MultiAgentOrchestrator:
    def __init__(self, db: Database, llm: LLMGateway):
        self.db = db
        self.llm = llm
        self.pipeline: List[BaseAgent] = [
            PlannerAgent(llm),
            DataAnalystAgent(llm),
            ContentAgent(llm),
            CampaignAgent(llm),
            QAAgent(llm),
            ExecutorAgent(llm, db),
        ]

    def run_campaign(self, req: CampaignRequest) -> CampaignResponse:
        campaign_id = str(uuid.uuid4())
        ctx = AgentContext(
            request=req,
            campaign_id=campaign_id,
            users=self.db.fetch_users(),
        )

        if not ctx.users:
            raise HTTPException(status_code=400, detail="没有用户数据，请先初始化或导入用户数据")

        trace = []
        for agent in self.pipeline:
            started = datetime.utcnow()
            ctx = agent.run(ctx)
            trace.append(
                {
                    "agent": agent.name,
                    "started_at": started.isoformat(),
                    "finished_at": datetime.utcnow().isoformat(),
                }
            )

        summary = {
            "campaign_id": campaign_id,
            "agent_trace": trace,
            "plan": ctx.plan,
            "insights": ctx.insights,
            "approved_tasks": len([t for t in ctx.tasks or [] if t["status"] == TaskStatus.executed.value]),
            "rejected_tasks": len([t for t in ctx.tasks or [] if t["status"] == TaskStatus.rejected.value]),
        }

        return CampaignResponse(
            campaign_id=campaign_id,
            goal=req.goal,
            summary=summary,
            segments=ctx.segments or [],
            tasks=ctx.tasks or [],
            qa_result=ctx.qa_result or {},
        )


# ============================================================
# 7. FastAPI 应用
# ============================================================

db = Database(DB_PATH)
db.seed_demo_data()
llm = LLMGateway()
orchestrator = MultiAgentOrchestrator(db, llm)

app = FastAPI(
    title="Multi-Agent Ops Automation System",
    description="多 Agent 协同运营自动化系统：目标拆解、用户分群、文案生成、活动编排、风控审核、执行入队。",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "name": "Multi-Agent Ops Automation System",
        "status": "running",
        "docs": "/docs",
        "example_goal": "召回 30 天未活跃但历史消费超过 100 元的用户",
    }


@app.post("/campaigns/run", response_model=CampaignResponse)
def run_campaign(req: CampaignRequest):
    return orchestrator.run_campaign(req)


@app.get("/campaigns")
def list_campaigns():
    return db.list_campaigns()


@app.get("/tasks")
def list_tasks(campaign_id: Optional[str] = None):
    return db.list_tasks(campaign_id)


@app.post("/seed")
def seed():
    db.seed_demo_data()
    return {"ok": True, "message": "demo data seeded"}


# ============================================================
# 8. 本地命令行调试入口
# ============================================================

if __name__ == "__main__":
    req = CampaignRequest(
        goal="召回 30 天未活跃但历史消费超过 100 元的用户",
        channel=Channel.push,
        budget=5000,
        operator="demo_operator",
    )
    result = orchestrator.run_campaign(req)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
