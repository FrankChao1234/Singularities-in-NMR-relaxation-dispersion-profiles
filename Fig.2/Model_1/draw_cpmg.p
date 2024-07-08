set term postscript enhanced color
set output "A-B-C.ps"

set title ""
set xrange [0:1000]
set yrange [-800:0]
set zrange [10:20]
set ztics 2
set pm3d at s
set hidden3d front
set ticslevel 0
set isosamples 40
set cbrange [10:20]
splot 'data' with pm3d  

