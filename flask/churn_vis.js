//This is the javascript code that builds and updates the bar graph

window.updater = function(data) {
  //d3.select("#svg_container").text(data);
  my_data = data;
  console.log(data);

//    var svg_margin = { top: 20, right: 20, bottom: 20, left: 40 };
//    var svg_width = d3.select("body").node().getBoundingClientRect().width - svg_margin.left - svg_margin.right;
//    var svg_height = 300 - svg_margin.top - svg_margin.bottom;
//
//    var y = d3.scaleLinear()
//        .domain([0, d3.max(data, function(d) { return d.petal_length; })])
//        .range([svg_height, 0]);
//
//    var x = d3.scaleBand()
//        .domain(d3.range(data.length))
//        .range([0, svg_width])
//        .padding(0.1);
//
//    var species_list = d3.map(data, function (d) { return d.species;}).keys();
//
//    if (d3.select("#svg_container").select("svg").empty()) {
//
//
//        svg = d3.select("#svg_container").append("svg")
//          .attr("width", svg_width + svg_margin.left + svg_margin.right)
//          .attr("height", svg_height + svg_margin.top + svg_margin.bottom)
//          .append("g")
//          .attr("transform",
//              "translate(" + svg_margin.left + "," + svg_margin.top + ")");
//
//        svg.append("g")
//            .attr("transform", "translate(0," + svg_height + ")")
//            .attr("class", "x axis")
//            .call(d3.axisBottom(x));
//
//        // add the y Axis
//        svg.append("g")
//            .attr("class", "y axis")
//            .call(d3.axisLeft(y));
//    } else {
//        svg.attr("width", svg_width + svg_margin.left + svg_margin.right)
//        svg.selectAll("g.y.axis")
//            .call(d3.axisLeft(y));
//
//        svg.selectAll("g.x.axis")
//            .call(d3.axisBottom(x));
//    }
//
//    // DATA JOIN
//    // Join new data with old elements, if any.
//
//    var bars = svg.selectAll(".bar")
//        .data(data);
//
//    // UPDATE
//    // Update old elements as needed.
//
//    bars
//        .attr("style",function(d) { return "fill:" + d3.schemeCategory10[species_list.indexOf(d.species)];})
//        .attr("x", function(d, i) { return x(i); })
//        .attr("width", x.bandwidth())
//        .transition()
//        .duration(100)
//        .attr("y", function(d) { return y(d.petal_length); })
//        .attr("height", function(d) { return svg_height - y(d.petal_length); });
//
//    // ENTER + UPDATE
//    // After merging the entered elements with the update selection,
//    // apply operations to both.
//
//    bars.enter().append("rect")
//        .attr("class", "bar")
//        .attr("style",function(d) { return "fill:" + d3.schemeCategory10[species_list.indexOf(d.species)];})
//        .attr("x", function(d, i) { return x(i); })
//        .attr("width", x.bandwidth())
//        .attr("y", function(d) { return y(d.petal_length); })
//        .attr("height", function(d) { return svg_height - y(d.petal_length); })
//        .merge(bars);
//
//    // EXIT
//    // Remove old elements as needed.
//
//    bars.exit().remove();

};