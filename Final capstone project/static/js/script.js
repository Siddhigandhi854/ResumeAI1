document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const downloadBtn = document.getElementById('download-btn');
    let analysisData = null;

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        
        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            analysisData = data;
            displayResults(data);
            
            results.classList.remove('hidden');
            window.scrollTo({ top: results.offsetTop - 50, behavior: 'smooth' });
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });

    function displayResults(data) {
        // ML Prediction
        document.getElementById('predicted-role').textContent = data.predicted_role;
        document.getElementById('confidence-score').textContent = data.confidence_score;
        
        const top5List = document.getElementById('top-5-roles');
        top5List.innerHTML = '';
        data.top_5_roles.forEach(role => {
            const li = document.createElement('li');
            li.textContent = `${role.role} (${role.prob}%)`;
            top5List.appendChild(li);
        });

        // ATS Score
        document.getElementById('ats-score').textContent = data.ats_score;
        document.getElementById('skills-match').textContent = data.ats_details.skills_match;
        document.getElementById('keywords-score').textContent = data.ats_details.keywords;
        document.getElementById('experience-score').textContent = data.ats_details.experience;
        document.getElementById('education-score').textContent = data.ats_details.education;
        document.getElementById('structure-score').textContent = data.ats_details.structure;

        // Job Match
        document.getElementById('match-percentage').textContent = data.match_percentage;
        const matchLabel = document.getElementById('match-label');
        matchLabel.textContent = data.match_label;
        matchLabel.className = 'match-label ' + data.match_label.toLowerCase();

        // Strength
        const strengthLabel = document.getElementById('strength-label');
        strengthLabel.textContent = data.strength;
        strengthLabel.className = 'strength-label strength-' + data.strength;
        document.getElementById('regression-score').textContent = data.regression_score;

        // Skills
        const extractedSkills = document.getElementById('extracted-skills');
        extractedSkills.innerHTML = data.ats_details.found_skills.map(s => `<span>${s}</span>`).join('');
        
        const missingSkills = document.getElementById('missing-skills');
        missingSkills.innerHTML = data.ats_details.missing_skills.map(s => `<span>${s}</span>`).join('');
        
        document.getElementById('skill-gap').textContent = data.skill_gap_percentage;

        // Keywords
        const keywordList = document.getElementById('keyword-list');
        keywordList.innerHTML = Object.entries(data.keyword_density)
            .map(([kw, info]) => `<p><b>${kw}</b>: ${info.count} times (${info.density}%)</p>`)
            .join('');

        // Sections
        document.getElementById('found-sections').textContent = data.found_sections.join(', ');
        document.getElementById('missing-sections').textContent = data.missing_sections.join(', ');

        // AI Suggestions
        document.getElementById('ai-strengths').innerHTML = data.ai_suggestions.strengths.map(s => `<li>${s}</li>`).join('');
        document.getElementById('ai-weaknesses').innerHTML = data.ai_suggestions.weaknesses.map(s => `<li>${s}</li>`).join('');
        document.getElementById('ai-improvements').innerHTML = data.ai_suggestions.improvements.map(s => `<li>${s}</li>`).join('');
        
        document.getElementById('rec-projects').textContent = data.ai_suggestions.recommended_projects.join(', ');
        document.getElementById('rec-skills').textContent = data.ai_suggestions.skills_to_improve.join(', ');

        // Setup Navigation for Advanced Insights
        const targetRole = document.getElementById('target_role').value || data.predicted_role;
        const missingSkillsData = data.ats_details.missing_skills;

        document.getElementById('nav-courses').onclick = () => {
            const params = new URLSearchParams({ role: targetRole });
            missingSkillsData.forEach(s => params.append('missing_skills', s));
            window.location.href = `/courses?${params.toString()}`;
        };

        document.getElementById('nav-market').onclick = () => {
            window.location.href = `/market?role=${encodeURIComponent(targetRole)}`;
        };

        document.getElementById('nav-salary').onclick = () => {
            window.location.href = `/salary?role=${encodeURIComponent(targetRole)}`;
        };

        document.getElementById('nav-interview').onclick = () => {
            window.location.href = `/interview?role=${encodeURIComponent(targetRole)}`;
        };
    }

    downloadBtn.addEventListener('click', async () => {
        if (!analysisData) return;

        try {
            const response = await fetch('/download_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(analysisData)
            });

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'Resume_Analysis_Report.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (error) {
            console.error('Error downloading report:', error);
            alert('Failed to download report.');
        }
    });
});
