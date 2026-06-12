_DISTANCE_FORWARD = r'''
        float3 delta = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a2), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float {result_name} = len_f3(delta);
        if ({result_name} < 1e-12f) continue;
        float inv_r = 1.0f / {result_name};
'''

_DISTANCE_FORCE = r'''
        {{
            float _grad_{result_name} = {grad_expr};
            float f_common = _grad_{result_name} * inv_r;
            float3 fvec = scale_f3(delta, f_common);
            add_force(f_x,f_y,f_z, a1, fvec);
            add_force(f_x,f_y,f_z, a2, scale_f3(fvec, -1.0f));
        }}
'''

_DISTANCE_13_FORWARD = r'''
        float3 r13v = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float {result_name} = len_f3(r13v);
        float inv_l13 = 0.0f;
        if ({result_name} >= 1e-12f) {{
            inv_l13 = 1.0f / {result_name};
        }}
'''

_DISTANCE_13_FORCE = r'''
        {{
            if ({result_name} >= 1e-12f) {{
                float _grad_{result_name} = {grad_expr};
                float f_ub = _grad_{result_name} * inv_l13;
                float3 f13 = scale_f3(r13v, f_ub);
                add_force(f_x,f_y,f_z, a1, f13);
                add_force(f_x,f_y,f_z, a3, scale_f3(f13, -1.0f));
            }}
        }}
'''

_ANGLE_FORWARD = r'''
        float3 r21 = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a1), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float3 r23 = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float l21 = len_f3(r21);
        float l23 = len_f3(r23);
        if (l21 < 1e-12f || l23 < 1e-12f) continue;
        float inv_l21 = 1.0f / l21;
        float inv_l23 = 1.0f / l23;
        float ct = dot_f3(r21, r23) * inv_l21 * inv_l23;
        ct = fmaxf(-1.0f, fminf(1.0f, ct));
        float {result_name} = acosf(ct);
'''

_ANGLE_FORCE = r'''
        {{
            float neg_dEdtheta = -({grad_expr});
            float3 n = cross_f3(r21, r23);
            float3 c1 = cross_f3(r21, n);
            float lc1 = len_f3(c1);
            if (lc1 > 1e-12f) {{
                float inv = neg_dEdtheta / (lc1 * l21);
                float3 fv1 = scale_f3(c1, inv);
                add_force(f_x,f_y,f_z, a1, fv1);
                add_force(f_x,f_y,f_z, a2, scale_f3(fv1, -1.0f));
            }}
            float3 c3 = cross_f3(scale_f3(r23, -1.0f), n);
            float lc3 = len_f3(c3);
            if (lc3 > 1e-12f) {{
                float inv = neg_dEdtheta / (lc3 * l23);
                float3 fv3 = scale_f3(c3, inv);
                add_force(f_x,f_y,f_z, a3, fv3);
                add_force(f_x,f_y,f_z, a2, scale_f3(fv3, -1.0f));
            }}
        }}
'''

_DIHEDRAL_FORWARD = r'''
        float3 rab = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a2), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float3 rbc = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float3 rcd = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a4), load_pos(pos_x,pos_y,pos_z,a3)), pbc_inv, pbc_matrix);
        float lab = len_f3(rab), lbc = len_f3(rbc), lcd = len_f3(rcd);
        if (lab < 1e-12f || lbc < 1e-12f || lcd < 1e-12f) continue;
        float3 n1 = cross_f3(rab, rbc);
        float3 n2 = cross_f3(rbc, rcd);
        float dn = dot_f3(n1, n2);
        float drn = dot_f3(rab, n2);
        float {result_name} = atan2f(lbc * drn, dn);
        float n1s = dot_f3(n1, n1);
        float n2s = dot_f3(n2, n2);
        if (n1s < 1e-12f || n2s < 1e-12f) continue;
'''

_DIHEDRAL_FORCE = r'''
        {{
            float fv = -({grad_expr});
            float fa = fv * lbc / n1s;
            float fd = fv * lbc / n2s;
            float3 f_a = scale_f3(n1, -fa);
            float3 f_d = scale_f3(n2, fd);
            float3 voc = scale_f3(rbc, 0.5f);
            float loc = lbc * 0.5f;
            float ils = 1.0f / (loc * loc);
            float3 t1 = cross_f3(voc, f_d);
            float3 t2 = scale_f3(cross_f3(rcd, f_d), 0.5f);
            float3 t3 = scale_f3(cross_f3(scale_f3(rab, -1.0f), f_a), 0.5f);
            float3 st = scale_f3(add_f3(t1, add_f3(t2, t3)), -1.0f);
            float3 f_c = scale_f3(cross_f3(st, voc), ils);
            float3 f_b = scale_f3(add_f3(f_a, add_f3(f_c, f_d)), -1.0f);
            add_force(f_x,f_y,f_z, a1, f_a);
            add_force(f_x,f_y,f_z, a2, f_b);
            add_force(f_x,f_y,f_z, a3, f_c);
            add_force(f_x,f_y,f_z, a4, f_d);
        }}
'''

HELPER_REGISTRY = {
    'distance': {
        'n_args': 2,
        'arg_indices': [0, 1],
        'forward': _DISTANCE_FORWARD,
        'force': _DISTANCE_FORCE,
    },
    'distance_13': {
        'n_args': 2,
        'arg_indices': [0, 2],
        'forward': _DISTANCE_13_FORWARD,
        'force': _DISTANCE_13_FORCE,
    },
    'angle': {
        'n_args': 3,
        'arg_indices': [0, 1, 2],
        'forward': _ANGLE_FORWARD,
        'force': _ANGLE_FORCE,
    },
    'dihedral': {
        'n_args': 4,
        'arg_indices': [0, 1, 2, 3],
        'forward': _DIHEDRAL_FORWARD,
        'force': _DIHEDRAL_FORCE,
    },
}
