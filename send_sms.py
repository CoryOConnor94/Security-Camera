from twilio.rest import Client
import time

# ------------------------------------Text via twilio----------------------------------------------
account_sid = ""
auth_token = ""

client = Client(account_sid, auth_token)

message = client.api.account.messages.create(
    to="",
    from_="",
    body=""
)

