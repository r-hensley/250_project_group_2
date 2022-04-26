
From https://supernovae.in2p3.fr/sdss_snls_jla/ReadMe.html
```
name: name of the SN
zcmb: CMB frame redshift (including peculiar velocity corrections for
      nearby supernova based on the models of M.J. Hudson)
zhel: Heliocentric redshift (note both zcmb and zhel are needed
      to compute the luminosity distance)
dz: redshift error (no longer used by the plugin)
mb: B band peak magnitude
dmb: Error in mb (includes contributions from intrinsic dispersion, 
     lensing, and redshift uncertainty)
```
     
The following parameters are in the file but are unused

```
x1: SALT2 shape parameter
dx1: Error in shape parameter
colour: Colour parameter
dcolour: Error in colour
3rdvar: In these files, the log_10 host stellar mass
d3rdvar: Error in 3rdvar
tmax: Date of peak brightness (mjd)
dtmax: Error in tmax
cov_m_s: The covariance between mb and x1
cov_m_c: The covariance between mb and colour
cov_s_c: The covariance between x1 and colour
set: A number indicating which sample this SN belongs to, with
   1 - SNLS, 2 - SDSS, 3 - low-z, 4 - Riess HST
ra: Right Ascension in degree (J2000)
dec: Declination in degree (J2000)
biascor: The correction for analysis bias applied to measured magnitudes
	 (this correction is already applied to mb, original measurements
	  can be obtained by subtracting this term to mb)
```
