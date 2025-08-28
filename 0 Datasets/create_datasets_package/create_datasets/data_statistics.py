class DataStatistics:
    def __init__(self, df):
        self.df = df

    def calculate_and_print_global_statistics(self):
        """
        Calculate and print global statistics for the dataset.
        """
        import tabulate as tabulate
        
        df_global_statistics = self.df.copy()

        if "train.TOP-DECOUPLED" not in df_global_statistics.columns:
            raise ValueError("The column 'train.TOP-DECOUPLED' does not exist in the dataset.")

        df_global_statistics["PIZZAORDER_count"] = df_global_statistics["train.TOP-DECOUPLED"].str.count("PIZZAORDER")
        df_global_statistics["DRINKORDER_count"] = df_global_statistics["train.TOP-DECOUPLED"].str.count("DRINKORDER")
        df_global_statistics["TOTAL_ORDER_count"] = df_global_statistics["PIZZAORDER_count"] + df_global_statistics["DRINKORDER_count"]

        def calculate_statistics(column):
            return {
                "mean": column.mean(),
                "median": column.median(),
                "mode": column.mode()[0],
                "quantiles": column.quantile([0.25, 0.5, 0.75]).to_dict(),
                "std_dev": column.std(),
                "min": column.min(),
                "max": column.max()
            }

        pizza_stats = calculate_statistics(df_global_statistics["PIZZAORDER_count"])
        drink_stats = calculate_statistics(df_global_statistics["DRINKORDER_count"])
        total_stats = calculate_statistics(df_global_statistics["TOTAL_ORDER_count"])

        headers = ["Statistic", "Pizza Order", "Drink Order", "Total Order"]
        rows = [
            ["Mean", pizza_stats["mean"], drink_stats["mean"], total_stats["mean"]],
            ["Median", pizza_stats["median"], drink_stats["median"], total_stats["median"]],
            ["Mode", pizza_stats["mode"], drink_stats["mode"], total_stats["mode"]],
            ["25th Percentile", pizza_stats["quantiles"][0.25], drink_stats["quantiles"][0.25], total_stats["quantiles"][0.25]],
            ["50th Percentile (Median)", pizza_stats["quantiles"][0.5], drink_stats["quantiles"][0.5], total_stats["quantiles"][0.5]],
            ["75th Percentile", pizza_stats["quantiles"][0.75], drink_stats["quantiles"][0.75], total_stats["quantiles"][0.75]],
            ["Standard Deviation", pizza_stats["std_dev"], drink_stats["std_dev"], total_stats["std_dev"]],
            ["Min", pizza_stats["min"], drink_stats["min"], total_stats["min"]],
            ["Max", pizza_stats["max"], drink_stats["max"], total_stats["max"]]
        ]

        print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))