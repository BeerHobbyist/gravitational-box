## Running the project
To build the project use the following commands:
```
mkdir build && cd build
cmake ..
make
```
This should build the project which can be run using:
```
./gravbox_gpu # For gpu version
./gravbox_cpu # For cpu version
```

## Perormance comparison
For the gpu (RTX 5060 Ti) I got the following results:
- around 1000 fps when particles are still falling down (less collisions) which drops to around 45 - 50 fps when particles settle on the floor which forces more collision calculations.
For the cpu (Ryzen 5 9600X):
- at most 10 fps which instantly drops to 2 fps and then after the particles settle on the flor I get 0.2 fps and could barely close the window.

The gpu version also supports simple window resizing.
