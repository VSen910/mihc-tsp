#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <omp.h>

/* Euclidean distance calculation */
long distD(int i,int j,float *x,float*y)
{
	float dx=x[i]-x[j];
	float dy=y[i]-y[j]; 
	return(sqrtf( (dx*dx) + (dy*dy) ));
}

long nn_init(int *route, long cities, float *posx, float *posy) {
	// srand(time(NULL));
	int city = 0;
	route[0] = city;
	int *visited = (int*) calloc(cities, sizeof(int));
	visited[city] = 1;
	long dist = 0;

	for(int i=1; i<cities; i++) {
		int prevCity = route[i-1];
		int nextCity;

		long minDist = INT_MAX;
		for(int currCity=0; currCity<cities; currCity++) {
			long currDist = distD(prevCity, currCity, posx, posy);
			if(minDist > currDist && !visited[currCity]) {
				minDist = currDist;
				nextCity = currCity;
			}
		}

		dist += minDist;
		route[i] = nextCity;
		visited[nextCity] = 1;
	}
	free(visited);
	dist += distD(route[0], route[cities-1], posx, posy);
	return dist;
}


typedef struct {
	long minDist;
	int nextCity;
} MinDistInfo;

#pragma omp declare reduction(minimum : MinDistInfo : \
	(omp_out = (omp_in.minDist < omp_out.minDist || (omp_in.minDist == omp_out.minDist && omp_in.nextCity < omp_out.nextCity)) ? omp_in : omp_out)) \
	initializer(omp_priv = {LONG_MAX, -1})

long nn_init_parallel1(int *route, long cities, float *posx, float *posy) {
  route[0] = 0;
  int *visited = (int*) calloc(cities, sizeof(int));
  visited[0] = 1;
  long dist = 0;

  for(int i = 1; i < cities; i++) {
      int prevCity = route[i - 1];
      int nextCity = -1;
      long minDist = LONG_MAX;

			MinDistInfo minDistInfo = {minDist, nextCity};

      #pragma omp parallel for reduction(minimum:minDistInfo)
      for(int currCity = 0; currCity < cities; currCity++) {
        if (!visited[currCity]) {
          long currDist = distD(prevCity, currCity, posx, posy);
					MinDistInfo localMinDistInfo = {currDist, currCity};
          if (currDist < minDistInfo.minDist) {
            minDistInfo = localMinDistInfo;
          }
        }
      }

      dist += minDistInfo.minDist;
      route[i] = minDistInfo.nextCity;
      visited[minDistInfo.nextCity] = 1;
  }

  free(visited);
  dist += distD(route[0], route[cities - 1], posx, posy);
  return dist;
}

long nn_init_parallel(int *route, long cities, float *posx, float *posy) {
    route[0] = 0;
    int *visited = (int*) calloc(cities, sizeof(int));
    visited[0] = 1;
    long dist = 0;

    for(int i = 1; i < cities; i++) {
        int prevCity = route[i - 1];
        int nextCity = -1;
        long minDist = LONG_MAX;

        #pragma omp parallel
        {
            int localNextCity = -1;
            long localMinDist = LONG_MAX;

            #pragma omp for nowait
            for(int currCity = 0; currCity < cities; currCity++) {
                if (!visited[currCity]) {
                    long currDist = distD(prevCity, currCity, posx, posy);
                    if (currDist < localMinDist) {
                        localMinDist = currDist;
                        localNextCity = currCity;
                    }
                }
            }

            #pragma omp critical
            {
                if (localMinDist < minDist || (localMinDist == minDist && localNextCity < nextCity)) {
                    minDist = localMinDist;
                    nextCity = localNextCity;
                }
            }
        }

        dist += minDist;
        route[i] = nextCity;
        visited[nextCity] = 1;
    }

    free(visited);
    dist += distD(route[0], route[cities - 1], posx, posy);
    return dist;
}


void routeChecker(long N,int *r)
{
	int *v,i,flag=0;
	v=(int*)calloc(N,sizeof(int));	

	for(i=0;i<N;i++)
		v[r[i]]++;
	for(i=0;i<N;i++)
	{
		if(v[i] != 1 )
		{
			flag=1;
			printf("breaking at %d",i);
			break;
		}
	}
	if(flag==1)
		printf("\nroute is not valid");
	else
		printf("\nroute is valid");
}

/* Arrange coordinate in initial solution's order*/
void setCoord(int *r,float *posx,float *posy,float *px,float *py,long cities)
{
	int i;
	for(i=0;i<cities;i++)
	{
		px[i]=posx[r[i]];
		py[i]=posy[r[i]];
	}
}

void isSame(int *r, int *r_p, int cities) {
  for(int i=0; i<cities; i++) {
    if(r[i] != r_p[i]) {
      printf("\nRoutes not same from index: %d\n", i);
      return;
    }
  }
  printf("\nRoutes same\n");
}

int main(int argc, char *argv[]) {
  int ch, cnt, in1;
	float in2, in3;
	FILE *f;
	float *posx, *posy;
	float *px, *py,tm;
	char str[256];  
	int *r;
  int *r_p;
	long sol,d,cities,no_pairs,tid=0;
	long dst;
	long intl,count = 0;
	
	clock_t start,end,start1,end1;

	f = fopen(argv[1], "r");
	if (f == NULL) {fprintf(stderr, "could not open file \n");  exit(-1);}

	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

	ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
	fscanf(f, "%s\n", str);
	cities = atoi(str);
	if (cities <= 2) {fprintf(stderr, "only %ld cities\n", cities);  exit(-1);}

	sol=cities*(cities-1)/2;
	posx = (float *)malloc(sizeof(float) * cities);  if (posx == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
	posy = (float *)malloc(sizeof(float) * cities);  if (posy == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
	px = (float *)malloc(sizeof(float) * cities);  if (px == NULL) {fprintf(stderr, "cannot allocate posx\n");  exit(-1);}
	py = (float *)malloc(sizeof(float) * cities);  if (py == NULL) {fprintf(stderr, "cannot allocate posy\n");  exit(-1);}
	
	r = (int *)malloc(sizeof(int) * cities);
	r_p = (int *)malloc(sizeof(int) * cities);
	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
	fscanf(f, "%s\n", str);
	if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

	cnt = 0;

	while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) 
	{
		posx[cnt] = in2;
		posy[cnt] = in3;
		cnt++;
		if (cnt > cities) {fprintf(stderr, "input too long\n");  exit(-1);}
		if (cnt != in1) {fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);  exit(-1);}
	}

	if (cnt != cities) {fprintf(stderr, "read %d instead of %ld cities\n", cnt, cities);  exit(-1);}
	fscanf(f, "%s", str);
	if (strcmp(str, "EOF") != 0) {fprintf(stderr, "didn't see 'EOF' at end of file\n");  exit(-1);}

	double dtime;

  dtime = omp_get_wtime();

	dst = nn_init(r,cities,posx,posy);
	routeChecker(cities, r);
	setCoord(r,posx,posy,px,py,cities);

  dtime = omp_get_wtime() - dtime;

	// printf("\nSequential");
	// printf("\nRoute:");
	// for(int i=0; i<cities; i++) {
	// 	printf("%d->", r[i]);
	// }
	// printf("%d", r[0]);
	printf("\ninitial cost : %ld time : %f\n",dst,dtime);

  // dtime = omp_get_wtime();

	// dst = nn_init_parallel1(r_p,cities,posx,posy);
	// routeChecker(cities, r_p);
	// setCoord(r_p,posx,posy,px,py,cities);

  // dtime = omp_get_wtime() - dtime;

	// printf("\nParallel");
	// printf("\nRoute:");
	// for(int i=0; i<cities; i++) {
	// 	printf("%d->", r_p[i]);
	// }
	// printf("%d", r_p[0]);
	// printf("\ninitial cost : %ld time : %f\n",dst,dtime);

  // isSame(r, r_p, cities);

	// dtime = omp_get_wtime();

	// long currDist = dst;
	// int x, y;
	
	// do {
	// 	dst = currDist;

	// 	for(int i=0; i<cities-1; i++) {
	// 		for(int j=i+1; j<cities; j++) {
	// 			long newDist = dst;
	// 			long change = distD(i, j, px, py) 
	// 			+ distD(i+1, (j+1)%cities, px, py) 
	// 			- distD(i, i+1, px, py) 
	// 			- distD(j, (j+1)%cities, px, py);
	// 			newDist += change;
	// 			if(newDist < currDist) {
	// 				x = i;
	// 				y = j;
	// 				currDist = newDist;
	// 			}
	// 		}
	// 	}

	// 	if(currDist < dst) {
	// 		float *tmp_x,*tmp_y;
	// 		int i, j;
	// 		tmp_x=(float*)malloc(sizeof(float)*(y-x));	
	// 		tmp_y=(float*)malloc(sizeof(float)*(y-x));	
	// 		for(j=0,i=y;i>x;i--,j++) {
	// 			tmp_x[j]=px[i];
	// 			tmp_y[j]=py[i];
	// 		}
	// 		for(j=0,i=x+1;i<=y;i++,j++) {
	// 			px[i]=tmp_x[j];
	// 			py[i]=tmp_y[j];
	// 		}
	// 		free(tmp_x);
	// 		free(tmp_y);
	// 	}
	// 	count++;
	// } while(currDist < dst);

	// dtime = omp_get_wtime() - dtime;

	// Parallel iterative hill climbing
	dtime = omp_get_wtime();

	long currDist = dst;
	int x, y;
	
	do {
		dst = currDist;

		#pragma omp parallel
		{
			int x_local;
			int y_local;
			long currDist_local = dst;

			#pragma omp for schedule(dynamic, 1)
			for(int i=0; i<cities-1; i++) {
				for(int j=i+1; j<cities; j++) {
					long newDist = dst;
					long change = distD(i, j, px, py) 
					+ distD(i+1, (j+1)%cities, px, py) 
					- distD(i, i+1, px, py) 
					- distD(j, (j+1)%cities, px, py);
					newDist += change;
					if(newDist < currDist_local) {
						x_local = i;
						y_local = j;
						currDist_local = newDist;
					}
				}
			}

			#pragma omp critical
			{
				if(currDist_local < currDist || (currDist_local == currDist && x_local < x) || (currDist_local == currDist && x_local == x && y_local < y)) {
					x = x_local;
					y = y_local;
					currDist = currDist_local;
				}
			}
		}

		if(currDist < dst) {
			float *tmp_x,*tmp_y;
			int i, j;
			tmp_x=(float*)malloc(sizeof(float)*(y-x));	
			tmp_y=(float*)malloc(sizeof(float)*(y-x));	
			for(j=0,i=y;i>x;i--,j++) {
				tmp_x[j]=px[i];
				tmp_y[j]=py[i];
			}
			for(j=0,i=x+1;i<=y;i++,j++) {
				px[i]=tmp_x[j];
				py[i]=tmp_y[j];
			}
			free(tmp_x);
			free(tmp_y);
		}
		count++;
	} while(currDist < dst);

  printf("\nMinimal distance found %ld\n",dst);
  printf("\nnumber of time hill climbed %ld\n",count);
  // end1 = clock();
	dtime = omp_get_wtime() - dtime;
  // printf("\ntime : %f\n",((double) (end1 - start1)) / CLOCKS_PER_SEC);
	printf("\ntime : %f\n",dtime);
  free(posx);
  free(posy);
  return 0;

}