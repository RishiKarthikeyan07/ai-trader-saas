"""
Alert System

Send email notifications via SendGrid
"""

from typing import Dict
from app.core.config import settings
from app.core.supabase import supabase

# SendGrid integration (commented until configured)
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail


async def send_signal_ready_alert(signal: Dict):
    """
    Send alert when signal moves to READY

    1. Find users who want this signal type
    2. Send email via SendGrid
    3. Create alert record
    """

    # Create alert records for all Elite users
    # (In production, filter by user preferences)
    elite_users = await get_elite_users()

    for user in elite_users:
        await create_alert(
            user_id=user['user_id'],
            signal_id=signal['id'],
            type='signal_ready',
            message=f"{signal['symbol']} is now READY to trade"
        )

        # Send email
        await send_email(
            to_email=user['email'],
            subject=f"🎯 Signal Ready: {signal['symbol']}",
            content=format_signal_email(signal)
        )


async def send_email(to_email: str, subject: str, content: str):
    """Send email via SendGrid"""

    # Stub implementation
    print(f"[EMAIL] To: {to_email}")
    print(f"[EMAIL] Subject: {subject}")
    print(f"[EMAIL] Content: {content}")

    # TODO: Uncomment when SendGrid is configured
    # message = Mail(
    #     from_email=settings.FROM_EMAIL,
    #     to_emails=to_email,
    #     subject=subject,
    #     html_content=content
    # )
    #
    # sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
    # response = sg.send(message)


async def create_alert(user_id: str, signal_id: str, type: str, message: str):
    """Create alert record in database"""
    alert_data = {
        'user_id': user_id,
        'signal_id': signal_id,
        'type': type,
        'message': message,
        'read': False
    }

    supabase.table('alerts').insert(alert_data).execute()


async def get_elite_users():
    """Get all Elite tier users"""
    response = supabase.table('profiles') \
        .select('user_id, email') \
        .eq('tier', 'elite') \
        .execute()

    return response.data


def format_signal_email(signal: Dict) -> str:
    """Format signal as HTML email"""
    return f"""
    <html>
        <body style="font-family: Arial, sans-serif; background: #0f1419; color: #fff; padding: 20px;">
            <h1 style="color: #a855f7;">🎯 Signal Ready: {signal['symbol']}</h1>
            <p>Your signal is now ready to trade!</p>

            <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3>Entry Zone</h3>
                <p>₹{signal['entry_min']} - ₹{signal['entry_max']}</p>

                <h3>Stop Loss</h3>
                <p style="color: #ef4444;">₹{signal['stop_loss']}</p>

                <h3>Targets</h3>
                <p>TP1: ₹{signal['target_1']}<br>
                TP2: ₹{signal['target_2']}</p>
            </div>

            <p>Login to your AI Trader dashboard to view full details.</p>
        </body>
    </html>
    """
