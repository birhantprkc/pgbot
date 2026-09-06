package main

import (
	"context"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/pgrundev/pgbot/internal/config"
	"github.com/pgrundev/pgbot/internal/findings"
	"github.com/spf13/cobra"
)

// newConfigCmd groups the .pgbot.toml ergonomics: check, explain, init (B2-4).
func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Inspect and scaffold the .pgbot.toml suppression config",
		Long: "Work with pgbot's configuration file (.pgbot.toml): validate it, explain how it\n" +
			"would treat a finding, or scaffold one from a live run.",
	}
	cmd.AddCommand(newConfigCheckCmd(), newConfigExplainCmd(), newConfigInitCmd())
	return cmd
}

func newConfigCheckCmd() *cobra.Command {
	var path string
	cmd := &cobra.Command{
		Use:   "check",
		Short: "Validate the config and print the resolved settings (non-zero on any warning)",
		Long: "Loads the config the same way a run would, prints where it came from and every\n" +
			"resolved value, and lists warnings (unknown ids, bad thresholds, rules with no\n" +
			"expiry). Exits non-zero if there is any warning — wire it into CI.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runConfigCheck(path)
		},
	}
	cmd.Flags().StringVar(&path, "config", "", "path to .pgbot.toml (default: discover)")
	return cmd
}

func runConfigCheck(path string) error {
	cfg, err := config.Load(path)
	if err != nil {
		return err // fatal: unreadable --config or a credential-shaped key
	}
	src := cfg.Source
	if src == "" {
		src = "(none found — built-in defaults apply)"
	}
	fmt.Printf("config: %s\n", src)
	fmt.Printf("schema: %d\n\n", cfg.Schema)

	fmt.Println("thresholds (resolved):")
	def := findings.DefaultTunables()
	printThreshold("unused_index_min_size_mb", float64(cfg.Tunables.UnusedIndexMinBytes)/(1<<20), float64(def.UnusedIndexMinBytes)/(1<<20), cfg.ThresholdOverrides, cfg.Source)
	printThreshold("dead_ratio_warn", cfg.Tunables.DeadRatioWarn, def.DeadRatioWarn, cfg.ThresholdOverrides, cfg.Source)
	printThreshold("replica_lag_warn_seconds", cfg.Tunables.ReplicaLagWarnSec, def.ReplicaLagWarnSec, cfg.ThresholdOverrides, cfg.Source)

	if len(cfg.Severity) > 0 {
		fmt.Println("\nseverity remaps:")
		ids := make([]string, 0, len(cfg.Severity))
		for id := range cfg.Severity {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			fmt.Printf("  %-32s → %s\n", id, cfg.Severity[id])
		}
	}

	if len(cfg.Ignore) > 0 {
		fmt.Println("\nignore rules:")
		for _, r := range cfg.Ignore {
			exp := r.Expires
			if exp == "" {
				exp = "no expiry"
			}
			fmt.Printf("  %-40s [%s]\n", r.String(), exp)
		}
	}

	warnings := append([]string(nil), cfg.Warnings...)
	warnings = append(warnings, cfg.HygieneWarnings()...)
	if len(warnings) == 0 {
		fmt.Println("\n✓ no warnings")
		return nil
	}
	fmt.Printf("\n%d warning(s):\n", len(warnings))
	for _, w := range warnings {
		fmt.Fprintf(os.Stderr, "  ! %s\n", w)
	}
	return fmt.Errorf("%d config warning(s)", len(warnings))
}

func printThreshold(key string, resolved, def float64, overrides map[string]float64, source string) {
	if _, ok := overrides[key]; ok {
		fmt.Printf("  %-26s = %-10g (from %s)\n", key, resolved, source)
	} else {
		fmt.Printf("  %-26s = %-10g (default)\n", key, def)
	}
}

func newConfigExplainCmd() *cobra.Command {
	var path string
	cmd := &cobra.Command{
		Use:   "explain <finding_id> [object]",
		Short: "Show how the config would treat a finding (which rule fires, and why)",
		Args:  cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(path)
			if err != nil {
				return err
			}
			object := ""
			if len(args) == 2 {
				object = args[1]
			}
			fmt.Print(cfg.Explain(args[0], object, time.Now()))
			return nil
		},
	}
	cmd.Flags().StringVar(&path, "config", "", "path to .pgbot.toml (default: discover)")
	return cmd
}

func newConfigInitCmd() *cobra.Command {
	var f inspectFlags
	var out string
	cmd := &cobra.Command{
		Use:   "init [connection-string]",
		Short: "Scaffold a commented .pgbot.toml seeded from a live run's findings",
		Long: "Runs a read-only inspection and writes a .pgbot.toml where every finding is a\n" +
			"commented-out ignore rule, pre-filled with its object — uncomment the ones you've\n" +
			"reviewed. Prints to stdout by default; use --output to write a file (never\n" +
			"overwrites an existing one).",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runConfigInit(cmd, args, f, out)
		},
	}
	fl := cmd.Flags()
	fl.StringVarP(&out, "output", "o", "", "write to this path instead of stdout (refuses to overwrite)")
	fl.DurationVar(&f.interval, "interval", time.Second, "gap between the two counter samples")
	fl.IntVar(&f.ashHz, "ash-hz", 0, "active-session sampling rate in Hz (0 disables it; init doesn't need waits)")
	fl.DurationVar(&f.window, "window", time.Second, "active-session sampling window")
	f.noStore = true // scaffolding shouldn't perturb the baseline history
	return cmd
}

func runConfigInit(cmd *cobra.Command, args []string, f inspectFlags, out string) error {
	connString := firstNonEmpty(argAt(args, 0), os.Getenv("DATABASE_URL"), os.Getenv("PGBOT_DATABASE_URL"), pgServiceFallback())
	if connString == "" {
		return fmt.Errorf("no connection string (pass one or set $DATABASE_URL)")
	}
	f.noStore = true
	ctx, cancel := context.WithTimeout(cmd.Context(), 40*time.Second)
	defer cancel()

	c, _, err := gather(ctx, connString, f)
	if err != nil {
		return err
	}
	tmpl := config.InitTemplate(c.Findings, time.Now())

	if out == "" {
		fmt.Print(tmpl)
		return nil
	}
	if _, err := os.Stat(out); err == nil {
		return fmt.Errorf("%s already exists — refusing to overwrite (remove it or pick another --output)", out)
	}
	if err := os.WriteFile(out, []byte(tmpl), 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d finding(s) seeded as commented rules)\n", out, len(c.Findings))
	return nil
}
