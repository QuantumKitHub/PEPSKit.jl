# pepskit

This is some semi-public code (I use it - it doesn't work - there is no documentation - it's still public)

In this package we
    - define states
    - define operators
    - define environments
        - some are independent of the operator (boundary mpses,channels,...)
        - some are dependent on the operator (hamiltonian channels)
    - define algorithms

To minimize the amount of complexity/code we typically
    - define operations acting on the north (finding the north boundary, finding the north hamiltonian channels)
    - define how to rotate the peps such that other directions now become north
