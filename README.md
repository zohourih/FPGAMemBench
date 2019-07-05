# FPGAMemBench

Memory benchmark for Intel FPGAs to measure memory bandwidth of OpenCL-supported boards.
Supports different blocking shapes, block overlapping and padding.

# To enable Fmax Override:

**Quartus Prime Standard:**

- Backup \*install_dir\*/hld/ip/board/bsp/adjust_plls.tcl
- Edit as follows:

Replace

```
for {set i 0} {$i < $argc} {incr i} {
   set v [lindex $argv $i]

   if {[string compare $v "-fmax"] == 0 && $i < [expr $argc - 1]} {
      set k_fmax [lindex $argv [expr $i + 1]]
      post_message "Forcing kernel frequency to $k_fmax"
   } elseif {[string compare $v "-skipmif"] == 0} {
      set do_update_mif 0
   } elseif {[string compare $v "-skipasm"] == 0} {
      set do_asm 0
   } elseif {[string compare $v "-skipsta"] == 0} {
      set do_sta 0
   } elseif {[string compare $v "-testpll"] == 0} {
      set do_sta 0
      set do_asm 0
      set do_update_mif 0
      set do_plltest 1
      set k_fmax 100.0
   }
}

if {$k_fmax == -1} {
    set x [get_kernel_clks_and_fmax $k_clk_name $k_clk2x_name]
    set k_fmax       [ lindex $x 0 ]
    set fmax1        [ lindex $x 1 ]
    set k_clk_name_full   [ lindex $x 2 ]
    set fmax2        [ lindex $x 3 ]
    set k_clk2x_name_full [ lindex $x 4 ]
}

post_message "Kernel Fmax determined to be $k_fmax\n";
```

With:

```
if {$k_fmax == -1} {
    set x [get_kernel_clks_and_fmax $k_clk_name $k_clk2x_name]
    set k_fmax       [ lindex $x 0 ]
    set fmax1        [ lindex $x 1 ]
    set k_clk_name_full   [ lindex $x 2 ]
    set fmax2        [ lindex $x 3 ]
    set k_clk2x_name_full [ lindex $x 4 ]
}

post_message "Kernel Fmax determined to be $k_fmax\n";

for {set i 0} {$i < $argc} {incr i} {
   set v [lindex $argv $i]

   if {[string compare $v "-fmax"] == 0 && $i < [expr $argc - 1]} {
      set k_fmax [lindex $argv [expr $i + 1]]
      post_message "Forcing kernel frequency to $k_fmax"
   } elseif {[string compare $v "-skipmif"] == 0} {
      set do_update_mif 0
   } elseif {[string compare $v "-skipasm"] == 0} {
      set do_asm 0
   } elseif {[string compare $v "-skipsta"] == 0} {
      set do_sta 0
   } elseif {[string compare $v "-testpll"] == 0} {
      set do_sta 0
      set do_asm 0
      set do_update_mif 0
      set do_plltest 1
      set k_fmax 100.0
   }
}
```

**Quartus Prime Pro:**

- Backup \*install_dir\*/hld/ip/board/bsp/adjust_plls_a10.tcl for below v18.1 and \*install_dir\*/hld/ip/board/bsp/adjust_plls.tcl for above.
- Edit as follows:

Replace
```
if {$k_fmax == -1} {
    set x [get_kernel_clks_and_fmax $k_clk_name $k_clk2x_name $iteration]
    set k_fmax       [ lindex $x 0 ]
    set fmax1        [ lindex $x 1 ]
    set k_clk_name_full   [ lindex $x 2 ]
    set fmax2        [ lindex $x 3 ]
    set k_clk2x_name_full [ lindex $x 4 ]
}

post_message "Kernel Fmax determined to be $k_fmax";
```

With:

```
if {$k_fmax == -1} {
    set x [get_kernel_clks_and_fmax $k_clk_name $k_clk2x_name $iteration]
    set k_fmax       [ lindex $x 0 ]
    set fmax1        [ lindex $x 1 ]
    set k_clk_name_full   [ lindex $x 2 ]
    set fmax2        [ lindex $x 3 ]
    set k_clk2x_name_full [ lindex $x 4 ]
}

post_message "Kernel Fmax determined to be $k_fmax";

for {set i 0} {$i < $argc} {incr i} {
   set v [lindex $argv $i]

   if {[string compare $v "-fmax"] == 0 && $i < [expr $argc - 1]} {
      set k_fmax [lindex $argv [expr $i + 1]]
      post_message "Forcing kernel frequency to $k_fmax"
   } elseif {[string compare $v "-skipmif"] == 0} {
      set do_update_mif 0
   } elseif {[string compare $v "-skipasm"] == 0} {
      set do_asm 0
   } elseif {[string compare $v "-skipsta"] == 0} {
      set do_sta 0
   } elseif {[string compare $v "-testpll"] == 0} {
      set do_sta 0
      set do_asm 0
      set do_update_mif 0
      set do_plltest 1
      set k_fmax 100.0
   }
}
```