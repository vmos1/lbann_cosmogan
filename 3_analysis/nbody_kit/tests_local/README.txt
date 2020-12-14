# Command

Command to run 3pct.py script

salloc -N 1 -p debug -t 00:30:00 -C haswell --image=nugent68/bccp:1.2
shifter /bin/bash
python get3pct.py --slice 64 -i 1 2 3 4 5 35 36 37 38 39 40