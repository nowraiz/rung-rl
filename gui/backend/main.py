from fastapi import FastAPI
from fastapi import Request, Response, WebSocket
import uuid

from sessions.session_manager import SessionManager

app = FastAPI()
session_manager = SessionManager()


@app.get("/")
def index(req: Request, res: Response):
    """
    Serve the index page html
    """
    return {"message": "Hello World", "Cookie": req.cookies.get("session_id", "No cookie")}


@app.get("/create")
def create_game(req: Request, res: Response):
    """
    Create a new single player game session
    """
    session_id = create_new_session(req, res)
    game = session_manager.create_single_player_game(session_id)
    game.signal_start()


@app.get("/join")
def join_game(req: Request, res: Response):
    """
    Creates a new multi-player game session. If no other player is available waits for a player to arrive.
    """
    session_id = create_new_session(req, res)
    # add the current player to the queue of waiting players
    # session_manager.add_to_queue(session_id)


@app.websocket("/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive()
        await websocket.send_text(f"Message text was: {data}")


def create_new_session(req: Request, res: Response):
    """
    Checks if a client has been assigned a session id, if so, returns it. Otherwise
    creates a new session id and returns it
    """
    session_id = req.cookies.get("session_id", None)

    if session_id is None:
        # a new player (ideally)
        session_id = str(uuid.uuid4())
        # set a new cookie if not present for the person
        res.set_cookie(key="session_id", value=session_id)

    return session_id
