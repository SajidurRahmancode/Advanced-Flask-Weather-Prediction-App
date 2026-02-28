import os, sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')

from app import create_app
from backend.models import db, User
from werkzeug.security import generate_password_hash

app, _ = create_app()
with app.app_context():
    users = User.query.all()
    print(f'Total users: {len(users)}')
    for u in users:
        print(f'  id={u.id}  username={u.username!r}  email={u.email!r}  active={u.is_active}')

    # Create admin112 user if it doesn't exist
    existing = User.query.filter(
        (User.email == 'admin112@gmail.com') | (User.username == 'admin112')
    ).first()
    if not existing:
        print("\nCreating admin112@gmail.com user...")
        u = User(username='admin112', email='admin112@gmail.com', is_active=True)
        u.set_password('admin112@gmail.com')
        db.session.add(u)
        db.session.commit()
        print(f"Created: id={u.id}  username={u.username}  email={u.email}")
    else:
        print(f"\nUser exists: id={existing.id}  email={existing.email}  active={existing.is_active}")
        # Reset password to ensure it matches
        existing.set_password('admin112@gmail.com')
        existing.is_active = True
        db.session.commit()
        print("Password reset to 'admin112@gmail.com'")
