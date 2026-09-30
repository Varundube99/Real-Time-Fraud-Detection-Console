import json, asyncio, logging, io
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import FraudShieldSystem, Config, TransactionBuilder, DEFAULT_GENUINE_V
from monitoring_service import MonitoringService

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
app = FastAPI(title='Console API', version='2.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    if any(request.url.path.endswith(ext) for ext in [".js", ".css", ".html"]):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

backend_system = FraudShieldSystem()
monitoring_service = MonitoringService(backend_system)
backend_system.monitor._seed_initial_data()

class ConfigRequest(BaseModel):
    speed: float = 1.0

class FraudRateRequest(BaseModel):
    fraud_rate: float = 0.05

@app.get('/api/stats')
def get_stats():
    snapshot = backend_system.dashboard()
    kpis = snapshot.get('kpis', {})
    hist_df = snapshot.get('history')
    alert_count = 0
    if hist_df is not None and not hist_df.empty:
        status_str = hist_df['Status'].astype(str).str.lower()
        alert_count = int((status_str.str.contains('blocked|fraud', regex=True)).sum())
    return {'kpis': kpis, 'is_streaming': monitoring_service.is_running, 'is_paused': monitoring_service.is_paused, 'alert_count': alert_count}

@app.get('/api/transactions')
def get_transactions(limit: int = 0):
    snapshot = backend_system.dashboard()
    hist_df = snapshot.get('history')
    if hist_df is None or hist_df.empty: return []
    records = hist_df.iloc[::-1].to_dict(orient='records')
    if limit > 0:
        return records[:limit]
    return records

@app.get('/api/transaction/{tx_id}')
def get_transaction_detail(tx_id: str):
    monitor = backend_system.monitor
    if hasattr(monitor, 'history') and monitor.history.history:
        for tx in reversed(monitor.history.history):
            if tx.transaction_id == tx_id:
                return {
                    "transaction_id": tx.transaction_id,
                    "timestamp": tx.timestamp,
                    "amount": tx.amount,
                    "time": tx.time,
                    "merchant": getattr(tx, "merchant", "Retail Merchant"),
                    "location": getattr(tx, "location", "Mumbai, IN"),
                    "device": getattr(tx, "device", "Mobile"),
                    "prediction": tx.prediction,
                    "probability": round(tx.probability * 100, 2),
                    "confidence": round(tx.confidence * 100, 2),
                    "latency_ms": tx.latency_ms,
                    "risk_level": tx.risk_level,
                    "decision": tx.decision,
                    "status": tx.decision,
                    "features": tx.features
                }
    raise HTTPException(status_code=404, detail="Transaction not found")

@app.get('/api/alerts')
def get_alerts(limit: int = 0):
    snapshot = backend_system.dashboard()
    hist_df = snapshot.get('history')
    if hist_df is None or hist_df.empty: return []
    import pandas as pd
    prob_num = pd.to_numeric(hist_df['Probability'].astype(str).str.replace('%', ''), errors='coerce').fillna(0.0)
    status_str = hist_df['Status'].astype(str).str.lower()
    alerts_df = hist_df[(status_str.str.contains('blocked|fraud', regex=True)) | (prob_num >= 50.0)]
    if alerts_df.empty: return []
    records = alerts_df.iloc[::-1].to_dict(orient='records')
    if limit > 0:
        return records[:limit]
    return records

@app.get('/api/export/alerts')
def export_alerts():
    snapshot = backend_system.dashboard()
    hist_df = snapshot.get('history')
    output = io.StringIO()
    if hist_df is not None and not hist_df.empty:
        import pandas as pd
        prob_num = pd.to_numeric(hist_df['Probability'].astype(str).str.replace('%', ''), errors='coerce').fillna(0.0)
        status_str = hist_df['Status'].astype(str).str.lower()
        alerts_df = hist_df[(status_str.str.contains('blocked|fraud', regex=True)) | (prob_num >= 50.0)]
        if not alerts_df.empty:
            alerts_df.iloc[::-1].to_csv(output, index=False)
    csv_data = output.getvalue()
    return Response(content=csv_data, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=fraud_alerts.csv'})

@app.get('/api/export/transactions')
def export_transactions():
    snapshot = backend_system.dashboard()
    hist_df = snapshot.get('history')
    output = io.StringIO()
    if hist_df is not None and not hist_df.empty:
        hist_df.iloc[::-1].to_csv(output, index=False)
    csv_data = output.getvalue()
    return Response(content=csv_data, media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=transaction_history.csv'})

@app.post('/api/control/speed')
def set_speed(req: ConfigRequest):
    if hasattr(backend_system, "monitor") and hasattr(backend_system.monitor, "state"):
        backend_system.monitor.state.speed = req.speed
    return {'status': 'success', 'speed': req.speed}

@app.post('/api/control/fraud_rate')
def set_fraud_rate(req: FraudRateRequest):
    monitor = getattr(backend_system, "monitor", None)
    if monitor and hasattr(monitor, "generator"):
        monitor.generator.set_fraud_rate(req.fraud_rate)
    return {'status': 'success', 'fraud_rate': req.fraud_rate}

@app.post('/api/control/{action}')
def control_stream(action: str):
    act = action.lower()
    if act == 'start': monitoring_service.start()
    elif act == 'pause': monitoring_service.pause()
    elif act == 'resume': monitoring_service.resume()
    elif act == 'stop': monitoring_service.stop()
    elif act == 'reset':
        monitoring_service.reset()
    else: raise HTTPException(status_code=400, detail=f'Invalid action {action}')
    return {'status': 'success', 'action': act, 'is_streaming': monitoring_service.is_running, 'is_paused': monitoring_service.is_paused}

@app.get('/api/stream/sse')
async def sse_stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected(): break
            snapshot = backend_system.dashboard()
            kpis = snapshot.get('kpis', {})
            hist_df = snapshot.get('history')
            recent_tx = []
            all_tx = []
            alert_count = 0
            recent_alerts = []
            all_alerts = []
            if hist_df is not None and not hist_df.empty:
                all_tx = hist_df.iloc[::-1].to_dict(orient='records')
                recent_tx = all_tx[:10]
                import pandas as pd
                prob_num = pd.to_numeric(hist_df['Probability'].astype(str).str.replace('%', ''), errors='coerce').fillna(0.0)
                status_str = hist_df['Status'].astype(str).str.lower()
                alerts_df = hist_df[(status_str.str.contains('blocked|fraud', regex=True)) | (prob_num >= 50.0)]
                alert_count = len(alerts_df)
                if not alerts_df.empty:
                    all_alerts = alerts_df.iloc[::-1].to_dict(orient='records')
                    recent_alerts = all_alerts[:5]
            
            payload = {
                'kpis': kpis,
                'is_streaming': monitoring_service.is_running,
                'is_paused': monitoring_service.is_paused,
                'alert_count': alert_count,
                'recent_transactions': recent_tx,
                'recent_alerts': recent_alerts,
                'all_transactions': all_tx,
                'all_alerts': all_alerts
            }
            yield f'data: {json.dumps(payload)}\n\n'
            await asyncio.sleep(1.0)
    return StreamingResponse(event_generator(), media_type='text/event-stream')

static_dir = Path(__file__).parent / 'static'
if static_dir.exists(): app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')

if __name__ == '__main__':
    import os, uvicorn
    port = int(os.environ.get('PORT', 8000))
    host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'
    uvicorn.run('server:app', host=host, port=port)

