document.addEventListener('DOMContentLoaded', function() {
    // Date filter toggle
    const useDateFilter = document.getElementById('useDateFilter');
    const dateRange = document.getElementById('dateRange');
    
    useDateFilter.addEventListener('change', function() {
        dateRange.style.display = this.checked ? 'block' : 'none';
    });

    // Example buttons
    document.querySelectorAll('.example-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.getElementById('searchInput').value = this.dataset.query;
            performSearch();
        });
    });

    // Search button
    document.getElementById('searchButton').addEventListener('click', performSearch);

    // Enter key in search input
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
});

async function performSearch() {
    const query = document.getElementById('searchInput').value;
    if (!query) return;

    const category = document.getElementById('category').value;
    const useDateFilter = document.getElementById('useDateFilter').checked;
    const startDate = useDateFilter ? document.getElementById('startDate').value : null;
    const endDate = useDateFilter ? document.getElementById('endDate').value : null;

    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                category,
                use_date_filter: useDateFilter,
                start_date: startDate,
                end_date: endDate
            })
        });

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while searching. Please try again.');
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';

    // Display answer
    document.getElementById('answer').textContent = data.answer;

    // Display thought process
    const thoughtProcessContent = document.getElementById('thoughtProcessContent');
    thoughtProcessContent.innerHTML = data.thought_process
        .map(thought => `<p>${thought}</p>`)
        .join('');

    // Display context sufficiency
    const contextSufficiency = document.getElementById('contextSufficiency');
    contextSufficiency.className = `alert ${data.enough_context ? 'alert-success' : 'alert-warning'}`;
    contextSufficiency.textContent = data.enough_context ? '✅ Sufficient Context' : '⚠️ Insufficient Context';

    // Display raw results
    const rawResultsContent = document.getElementById('rawResultsContent');
    rawResultsContent.innerHTML = data.results
        .map((result, index) => `
            <div class="card mb-3">
                <div class="card-body">
                    <h5 class="card-title">Result ${index + 1}</h5>
                    <div class="row">
                        <div class="col-md-9">
                            <h6>Question & Answer:</h6>
                            <p>${result.content}</p>
                        </div>
                        <div class="col-md-3">
                            <h6>Metadata:</h6>
                            <p>Category: ${result.metadata.category || 'N/A'}</p>
                            <p>Created: ${result.metadata.created_at || 'N/A'}</p>
                            <p>Similarity: ${result.similarity.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
            </div>
        `)
        .join('');
} 