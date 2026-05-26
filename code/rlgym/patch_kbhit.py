# patch_kbhit.py
import sys
import types

# Crée un faux module kbhit qui ne plante pas sans TTY
class KBHit:
    def __init__(self):
        try:
            import termios, tty
            self.fd = sys.stdin.fileno()
            self.old_term = termios.tcgetattr(self.fd)
            self.new_term = self.old_term[:]
            tty.setraw(self.fd)
        except Exception:
            self.fd = None
            self.old_term = None
            self.new_term = None

    def kbhit(self):
        return False  # pas de clavier en mode headless

    def getch(self):
        return ''

    def set_normal_term(self):
        try:
            if self.old_term is not None:
                import termios
                termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
        except Exception:
            pass

# Injecte le faux module avant que rlgym_ppo l'importe
fake_module = types.ModuleType('rlgym_ppo.util.kbhit')
fake_module.KBHit = KBHit
sys.modules['rlgym_ppo.util.kbhit'] = fake_module