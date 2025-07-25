
import redis
import json 

# Store session ID that can be shared amoung different worker processes
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Function used to get/save/remove sessions
def get_session(session_id):
    if not session_id:
        return None
    session_json = r.get(session_id)
    return json.loads(session_json) if session_json else None

def save_session(session_id, session_data):
    r.set(session_id, json.dumps(session_data), ex=1800)  # expires in 30 min

def delete_session(session_id):
    r.delete(session_id)
