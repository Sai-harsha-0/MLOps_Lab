```mermaid
flowchart TD
	node1["data/raw/ratings.csv.dvc"]
	node2["features"]
	node3["load"]
	node1-->node3
	node3-->node2
```