import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch_mock = MagicMock()
torch_mock.cuda.is_available.return_value = False
sys.modules['torch'] = torch_mock
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.utils'] = MagicMock()
sys.modules['ollama'] = MagicMock()