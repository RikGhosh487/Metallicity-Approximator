/*
    Photometric Approximator of Stellar Metallicity (c) by Rik Ghosh, Soham Saha

    Photometric Approximator of Stellar Metallicity is licensed under a
    Creative Commons Attribution 4.0 International License.

    You should have received a copy of the license along with this work.
    If not, see https://creativecommons.org/licenses/by/4.0
*/

-- Collect color data and metallicity values for stars in the sppParams table of the SDSS DR 17
-- link: http://skyserver.sdss.org/dr17/SearchTools/SQL/
-- Note: Remove #Table from the .csv before obtaining training data (if present)

SELECT
    p.u - p.g as ug,                            -- color difference for u and g
    p.g - p.r as gr,                            -- color difference for g and r
    p.r - p.i as ri,                            -- color difference for r and i
    p.i - p.z as iz,                            -- color difference for i and z
    sp.fehadop as feh                           -- spectroscopic metallicity value
FROM sppParams AS sp
JOIN photoObj AS p ON p.objid = sp.bestobjid	-- merging SEGUE data with SDSS
WHERE
p.type = 6 and			                        -- stars only
p.mode = 1 and 			                        -- remove duplicates
sp.flag like 'nnnnn' and	                    -- normal SSPP flags
sp.fehdaopn >= 2 	                        	-- atleast 2 SSPP readings
