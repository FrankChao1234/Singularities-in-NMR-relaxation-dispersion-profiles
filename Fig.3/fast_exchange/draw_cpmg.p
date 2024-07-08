set term postscript enhanced color
set output "A-B-C.ps"

set title ""
set xrange [0:1000]
set yrange [-800:200]
set zrange [10:40]
set ztics 10
set pm3d at s
set hidden3d front
set ticslevel 0
set isosamples 40
splot 'data' using 1:($2+200):3 with pm3d  

