from fastapi import APIRouter

from app.core.state import set_kill_switch, get_state

router = APIRouter()


@router.post("/kill")
async def kill_switch():
    state = set_kill_switch(True)
    return {"kill_switch": state.kill_switch}


@router.get("/kill")
async def kill_status():
    state = get_state()
    return {"kill_switch": state.kill_switch}
