import smtplib
from email.message import EmailMessage
from config import settings


def _send(msg: EmailMessage):
    try:
        with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
            server.set_debuglevel(1)
            server.starttls()
            server.login(settings.EMAIL_USERNAME, settings.EMAIL_PASSWORD)
            server.send_message(msg)
            print("[email] ✅ Sent successfully")
    except Exception as e:
        print(f"[email] ❌ FAILED: {type(e).__name__}: {e}")
        raise


def send_admin_signup_notification(user_data: dict):
    """Notify admin when a new CS student registers."""
    msg = EmailMessage()
    msg["Subject"] = "New Student Signup — CS Tutor"
    msg["From"]    = settings.EMAIL_FROM
    msg["To"]      = settings.ADMIN_EMAIL

    msg.set_content(f"""
New student signup on CS Tutor:

Name:       {user_data['name']}
Email:      {user_data['email']}
Phone:      {user_data['phone']}
CS Level:   {user_data['cs_level']}
Attempt:    {user_data['cs_attempt']}
Plan:       {user_data.get('plan', 'free')}

Student is auto-approved and can log in immediately.
""")
    _send(msg)


def send_password_reset_otp(email: str, otp: str, name: str = "Student"):
    """Send a styled OTP email for password reset."""
    msg = EmailMessage()
    msg["Subject"] = "CS Tutor — Password Reset OTP"
    msg["From"]    = settings.EMAIL_FROM
    msg["To"]      = email

    msg.set_content(f"""
Hi {name},

You requested a password reset for your CS Tutor account.

Your OTP is:  {otp}

This OTP is valid for 10 minutes. Do not share it with anyone.

If you did not request this, please ignore this email.

— CS Tutor Team
""")

    msg.add_alternative(f"""\
<html>
  <body style="font-family: 'DM Sans', Arial, sans-serif; background: #f0f4ff; padding: 40px 0;">
    <div style="max-width: 480px; margin: 0 auto; background: #ffffff;
                border-radius: 16px; overflow: hidden;
                box-shadow: 0 4px 24px rgba(15,24,40,0.10);">

      <!-- Header -->
      <div style="background: #1c3557; padding: 32px 40px; text-align: center;">
        <div style="display: inline-block; background: #e8a020; color: #1c3557;
                    font-size: 22px; font-weight: 800; padding: 10px 22px;
                    border-radius: 10px; letter-spacing: 1px;">CS Tutor</div>
      </div>

      <!-- Body -->
      <div style="padding: 36px 40px;">
        <h2 style="color: #0f1828; font-size: 22px; margin: 0 0 8px;">Password Reset</h2>
        <p style="color: #4a5568; font-size: 15px; margin: 0 0 28px;">
          Hi {name}, use the OTP below to reset your password.
          It expires in <strong>10 minutes</strong>.
        </p>

        <!-- OTP Box -->
        <div style="background: #1c3557; border-radius: 12px;
                    padding: 28px; text-align: center; margin-bottom: 28px;">
          <div style="color: rgba(255,255,255,0.6); font-size: 12px;
                      letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;">
            Your OTP
          </div>
          <div style="color: #e8a020; font-size: 42px; font-weight: 800;
                      letter-spacing: 10px; font-family: monospace;">
            {otp}
          </div>
        </div>

        <p style="color: #8a96a8; font-size: 13px; text-align: center; margin: 0;">
          If you didn't request this, you can safely ignore this email.
        </p>
      </div>

      <!-- Footer -->
      <div style="background: #f0f4ff; padding: 20px 40px; text-align: center;">
        <p style="color: #8a96a8; font-size: 12px; margin: 0;">
          © CS Tutor · AI-Powered Learning for Company Secretaries · Secured
        </p>
      </div>
    </div>
  </body>
</html>
""", subtype="html")

    _send(msg)