from twilio.rest import Client
import cv2

account_sid = '[ENTER ACCOUNT SID]'
auth_token = '[ENTER AUTH TOKEN]'
client = Client(account_sid, auth_token)

test = cv2.imread("Original.jpg")

message = client.messages.create(
  from_='whatsapp:+141[ENTER WHATSAPP NUMBER]',
  body=f'Motion detected!',
  to='whatsapp:+353[ENTER PHONE NUMBER]'
)

print(message.sid)