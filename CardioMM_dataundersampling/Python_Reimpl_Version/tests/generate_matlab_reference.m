function generate_matlab_reference(toolbox_path, output_path)
addpath(toolbox_path);

% --- original cases ---
x = reshape(1:24, [4, 6]);
reference.zpad_even = zpad(x, [8, 10]);
reference.zpad_odd  = zpad(x, [7, 9]);
reference.crop_even = crop(reference.zpad_even, [4, 6]);
reference.crop_odd  = crop(reference.zpad_odd,  [4, 6]);

reference.uniform    = UniformSampling(32, 31, 8, 4);
reference.kt_uniform = ktUniformSampling(32, 31, 7, 8, 4);
reference.kt_gaussian = ktGaussianSampling(32, 31, 7, 8, 4, 0.2, 10);
reference.kt_radial   = ktRadialSampling(32, 31, 3, 8, 4.8, 137.5, true);

% --- 3D zpad / crop ---
x3 = reshape(1:24, [2, 3, 4]);
reference.zpad_3d = zpad(x3, [4, 6, 8]);
reference.crop_3d = crop(reference.zpad_3d, [2, 3, 4]);

% --- ktRadialSampling with cropcorner=false ---
reference.kt_radial_nocrop = ktRadialSampling(32, 31, 3, 8, 4.8, 137.5, false);

% --- ktMaskGenerator end-to-end (all four patterns) ---
reference.ktmask_uniform   = ktMaskGenerator(32, 31, 7, 8, 8, 'Uniform');
reference.ktmask_ktuniform = ktMaskGenerator(32, 31, 7, 8, 8, 'ktUniform');
% ktGaussian: call ktGaussianSampling directly with the same alpha/seed that
% ktMaskGenerator would use internally (alpha=0.2, seed=42).
reference.ktmask_ktgaussian = ktGaussianSampling(32, 31, 7, 8, 8, 0.2, 42);
% ktRadial: ktMaskGenerator applies R*0.6 = 8*0.6 = 4.8 before delegating.
reference.ktmask_ktradial  = ktMaskGenerator(32, 31, 3, 8, 8, 'ktRadial');

% --- nt=1 (4D kspace case, all four patterns) ---
reference.uniform_nt1     = ktMaskGenerator(32, 31, 1, 8, 8, 'Uniform');
reference.kt_uniform_nt1  = ktUniformSampling(32, 31, 1, 8, 4);
reference.kt_gaussian_nt1 = ktGaussianSampling(32, 31, 1, 8, 8, 0.2, 7);
reference.kt_radial_nt1   = ktRadialSampling(32, 31, 1, 8, 4.8, 137.5, true);

% --- randp direct reference ---
% Weighted draw: weight-2 bin (index 2) appears ~twice as often as the others.
reference.randp_weighted = randp([1.0, 2.0, 1.0], 42, 20, 1);
% Uniform draw over four bins — 2-D shape to exercise column-major vs row-major.
reference.randp_equal    = randp([1.0, 1.0, 1.0, 1.0], 7, 5, 3);

% --- ktdup direct reference ---
% One duplicate pair at (ph=0, ti=0); vacant slot in same frame at ph=1.
ph_in = [0; 0; 1; 2];
ti_in = [0; 0; 0; 1];
[ph_out, ti_out] = ktdup(ph_in, ti_in, 6, 4);
reference.ktdup_ph = ph_out;
reference.ktdup_ti = ti_out;

% --- UniformSampling with R=8 (second acceleration factor) ---
reference.uniform_r8 = UniformSampling(32, 31, 8, 8);

save(output_path, '-struct', 'reference', '-v7');
end
