import json
import sys

expected = ['argument with spaces', 'C:\\Program Files\\AMD\\', 'quote"inside']
received = sys.argv[1:]
print('CUDAAMD_LAUNCHER_ARGV:' + json.dumps({'expected': expected, 'received': received, 'pass': received == expected}, sort_keys=True))
if received != expected:
    raise SystemExit(1)
