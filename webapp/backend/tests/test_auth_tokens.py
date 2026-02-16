from app.services.auth import create_access_token, decode_access_token


def test_jwt_encode_decode_roundtrip():
    token = create_access_token({"sub": "alice", "role": "clinician"})
    payload = decode_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "clinician"
