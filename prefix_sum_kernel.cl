__kernel void prefix_sum(__global float* A, __global float* B, __global float* max_sum) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    __local float cache[64];
    cache[local_id] = A[global_id];

    for (int i = 1; i < 64; i <<= 1) {
        float sync = 0;
		if(i <= local_id){
			sync = cache[local_id - i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		cache[local_id] += sync;
    }
	
	barrier(CLK_LOCAL_MEM_FENCE);
    B[global_id] = cache[local_id];
	
	if(local_id == 63) {
		max_sum[get_group_id(0)] = cache[local_id];
	}
}

__kernel void total_prefix_sum(__global float* B, __global float* max_sum) {
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);
    uint groups_count = get_num_groups(0) - 1;

		for(int i = 0; i < groups_count; i++) {
			if(group_id <= i) {
				return;
			}
			B[global_id] += max_sum[i];
	}
}