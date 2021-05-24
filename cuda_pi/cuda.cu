#include <stdio.h>
#include <time.h>

long cantidadIntervalos = 1000000000;
double baseIntervalo;
double fdx;
double acum = 0;
clock_t start, end;

void main() {
   double x=0;
   long i;
   baseIntervalo = 1.0 / cantidadIntervalos;
   start = clock();
   for (i = 0; i < cantidadIntervalos; i++) {
      x = (i+0.5)*baseIntervalo;
      fdx = 4 / (1 + x * x);
      acum += fdx;
   }
   acum *= baseIntervalo;
   end = clock();
   printf("Result = %20.18lf (%ld)\n", acum, end - start);
}