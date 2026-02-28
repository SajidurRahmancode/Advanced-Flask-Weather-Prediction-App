import os, sys, logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, '.')

from app import create_app
from backend.models import db, User
from sqlalchemy import text

app, _ = create_app()
with app.app_context():
    # Widen column first
    try:
        db.session.execute(text("ALTER TABLE user MODIFY COLUMN password_hash VARCHAR(256) NOT NULL"))
        db.session.commit()
        print("Column widened to VARCHAR(256)")
    except Exception as e:
        print(f"ALTER note: {e}")
        db.session.rollback()

    # Find / create user
    u = User.query.filter(User.email == 'admin112@gmail.com').first()
    if not u:
        u = User()
        u.username = 'admin112@gmail.com'
        u.email = 'admin112@gmail.com'
        u.is_active = True
        db.session.add(u)

    u.set_password('admin112@gmail.com')
    u.is_active = True
    db.session.commit()
    ok = u.check_password('admin112@gmail.com')
    print(f"User id={u.id}  hash_len={len(u.password_hash)}  check_password={ok}")
