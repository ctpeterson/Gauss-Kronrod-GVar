# Gauss-Kronrod-GVar

Gauss-Kronrod integration compatible with Peter Lepage's GVar library (https://gvar.readthedocs.io/en/latest/). Simply create an integrator object as follows.

 ```
 import integrator as intg
 
 integrator = intg.NDimAdapGK()
  ```

Then integrate your favorite function as follows

  ```
  f = lambda x, y, z : x**2. * y**2. * z**2.
  ranges = [[.1, 3.], [2., 4.], [8., 10.]]
  result = integrator(f, ranges)
  ```
  
Caution: this code is still currently under development. Use with caution.
