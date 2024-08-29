/*
 * Copyright (C) 2024 Guillaume Perez
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>.
 * 
 * From the code of Laurent Condat: https://lcondat.github.io
 * 
*/

#ifndef PROJCODE_INCLUDE_UTILS_SORT_HPP
#define PROJCODE_INCLUDE_UTILS_SORT_HPP


#include <iostream>
#include <cstdio>

static void quicksort(double *z, int *z_perm, const int lo, const int hi) {
	int i=lo, j=hi, temp_id;
	double temp; 
	int pivot_id = lo+(int)(rand()/(((double)RAND_MAX+1.0)/(hi-lo+1)));
	double pivot_value = z[pivot_id];
	for (;;) {    
		while (z[i] > pivot_value) i++; 
		while (z[j] < pivot_value) j--;
		if (i >= j) break;
		temp_id = z_perm[i];
		temp = z[i];
		z_perm[i] = z_perm[j];
		z[i++]=z[j];
		z_perm[j] = temp_id;
		z[j--]=temp;
	}   
	if (i-1>lo) quicksort(z, z_perm, lo, i-1);
	if (hi>j+1) quicksort(z, z_perm, j+1, hi);
}


#endif /* PROJCODE_INCLUDE_UTILS_SORT_HPP */
