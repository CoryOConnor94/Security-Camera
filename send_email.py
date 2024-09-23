import smtplib

my_email = "[ENTER SENDER EMAIL]"
password = "[ENTER PASSWORD]"

# Create SMTP Object
with smtplib.SMTP("smtp.gmail.com") as connection:
    # Encrypt connection
    connection.starttls()
    # Login to connection
    connection.login(user=my_email, password=password)
    # Send email containing random quote
    connection.sendmail(from_addr=my_email,
                        to_addrs="[ENTER RECIPIENT EMAIL]",
                        msg=f"Subject:[ENTER MESSAGE]")

