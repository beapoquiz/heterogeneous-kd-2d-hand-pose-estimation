import shutil
total, used, free = shutil.disk_usage(r'C:\Users\Bea Juliana Poquiz\Desktop')
print(f'Free disk space: {free / (1024**3):.1f} GB')
print(f'Used: {used / (1024**3):.1f} GB')
print(f'Total: {total / (1024**3):.1f} GB')
print()
print('FreiHAND download size: ~3.5 GB')
print('Download link: https://lmb.informatik.uni-freiburg.de/projects/freihand/')